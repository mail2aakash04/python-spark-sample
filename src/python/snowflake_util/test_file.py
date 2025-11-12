#!/usr/bin/env python3
"""
Generic Schema Migration Analysis Tool (fixed)
Works with the provided snowflake_tables.csv and sole_tables.csv formats.
Usage: python3 generic_schema_analysis_fixed.py <product_name> <sf_file> <sole_file> [--output-dir OUTPUT_DIR]
"""

import pandas as pd
import requests
import json
from datetime import datetime
import argparse
import os
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher


# -----------------------
# Utility helpers
# -----------------------
def load_generic_schema_data(sf_file: str, sole_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generic function to load schema data from SF and SOLE files.
    Supports both CSV and Excel formats.
    """
    print(f"📊 Loading SOLE schema from: {sf_file}")
    print(f"📊 Loading SF schema from: {sole_file}")

    # Load SF data
    if sf_file.lower().endswith(('.xlsx', '.xls')):
        sf_data = pd.read_excel(sf_file)
    else:
        sf_data = pd.read_csv(sf_file, low_memory=False)

    # Load SOLE data
    if sole_file.lower().endswith(('.xlsx', '.xls')):
        sole_data = pd.read_excel(sole_file)
    else:
        sole_data = pd.read_csv(sole_file, low_memory=False)

    print(f"✅ SF data loaded: {len(sf_data)} rows")
    print(f"✅ SOLE data loaded: {len(sole_data)} rows")

    return sf_data, sole_data


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame: strip & lowercase column names and string values.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    # trim and lower string cells
    df = df.apply(lambda col: col.map(lambda x: x.strip().lower() if isinstance(x, str) else x))
    # df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
    return df


def detect_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Return the first column name from candidates that exists in df.columns.
    candidates should be lowercase names.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_objects_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build objects dict: {OBJECT_NAME_UPPER: {'type': TYPE, 'columns': [COL1, COL2, ...]}}
    Uses auto-detection of column names for object/type/column.
    """
    df_norm = normalize_dataframe(df)

    name_col = detect_column(df_norm, ['table_name', 'object_name', 'name'])
    type_col = detect_column(df_norm, ['table_type', 'object_type', 'type'])
    col_col = detect_column(df_norm, ['column_name', 'column', 'field_name'])

    if name_col is None:
        raise ValueError(
            "Could not detect object/table name column in dataframe. Columns found: " + ", ".join(df_norm.columns))

    objects: Dict[str, Dict[str, Any]] = {}

    for _, row in df_norm.iterrows():
        raw_name = row.get(name_col, None)
        if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
            continue
        obj_name = str(raw_name).strip().upper()
        obj_type = str(row.get(type_col, '')).strip().upper() if type_col else ''
        column_name = str(row.get(col_col, '')).strip().upper() if col_col else ''

        if not obj_name or obj_name == 'NAN':
            continue

        if obj_name not in objects:
            objects[obj_name] = {'type': obj_type, 'columns': []}
        if column_name and column_name != 'NAN':
            objects[obj_name]['columns'].append(column_name)

    return objects


# -----------------------
# Matching helpers
# -----------------------
def calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two object names using SequenceMatcher (0..1)."""
    if not name1 or not name2:
        return 0.0
    name1 = name1.upper().strip()
    name2 = name2.upper().strip()
    if name1 == name2:
        return 1.0
    return float(SequenceMatcher(None, name1, name2).ratio())


def check_type_compatibility(sf_type: str, sole_type: str) -> bool:
    """Check if SF and SOLE object types are compatible for migration."""
    sf_type = (sf_type or '').upper().strip()
    sole_type = (sole_type or '').upper().strip()

    # Direct match
    if sf_type and sole_type and sf_type == sole_type:
        return True

    # Enhanced compatibility: SOLE MANAGED = SF TABLE
    if sf_type in ['BASE TABLE', 'TABLE'] and sole_type in ['BASE TABLE', 'TABLE', 'MANAGED']:
        return True

    # Views matching
    if sf_type == 'VIEW' and sole_type == 'VIEW':
        return True

    # If either type is missing, don't block match (be permissive)
    if not sf_type or not sole_type:
        return True

    return False


def calculate_object_column_overlap(sf_object_info: Dict, sole_object_info: Dict) -> Dict[str, Any]:
    """
    Calculate column overlap and a set of stats between two objects.
    Returned dict contains keys used by downstream code.
    """
    sf_cols = set(c for c in (sf_object_info.get('columns') or []))
    sole_cols = set(c for c in (sole_object_info.get('columns') or []))

    # Use uppercase for comparisons (they should already be uppercase)
    sf_cols = set(c.upper() for c in sf_cols if c)
    sole_cols = set(c.upper() for c in sole_cols if c)

    common_cols = sf_cols.intersection(sole_cols)
    sf_only = sf_cols - sole_cols
    sole_only = sole_cols - sf_cols

    sf_column_count = len(sf_cols)
    sole_column_count = len(sole_cols)

    total_unique_cols = len(sf_cols.union(sole_cols))
    standard_overlap_ratio = (len(common_cols) / total_unique_cols) if total_unique_cols > 0 else 0.0
    sf_coverage_ratio = (len(common_cols) / sf_column_count) if sf_column_count > 0 else 0.0
    sole_coverage_ratio = (len(common_cols) / sole_column_count) if sole_column_count > 0 else 0.0

    if sole_column_count > sf_column_count:
        primary_overlap_ratio = sf_coverage_ratio
        matching_strategy = "sf_focused_ignore_sole_extras"
        match_quality = "high" if sf_coverage_ratio > 0.8 else "medium" if sf_coverage_ratio > 0.6 else "low"
    elif sf_column_count > sole_column_count:
        primary_overlap_ratio = sole_coverage_ratio
        matching_strategy = "sole_focused_ignore_sf_extras"
        match_quality = "high" if sole_coverage_ratio > 0.8 else "medium" if sole_coverage_ratio > 0.6 else "low"
    else:
        primary_overlap_ratio = standard_overlap_ratio
        matching_strategy = "standard_equal_columns"
        match_quality = "high" if standard_overlap_ratio > 0.8 else "medium" if standard_overlap_ratio > 0.6 else "low"

    return {
        'common_columns': len(common_cols),
        'sf_only_columns': len(sf_only),
        'sole_only_columns': len(sole_only),
        'sf_column_count': sf_column_count,
        'sole_column_count': sole_column_count,
        'overlap_ratio': primary_overlap_ratio,
        'sf_coverage_ratio': sf_coverage_ratio,
        'sole_coverage_ratio': sole_coverage_ratio,
        'standard_overlap_ratio': standard_overlap_ratio,
        'matching_strategy': matching_strategy,
        'match_quality': match_quality,
        'common_column_list': sorted(list(common_cols))
    }


# -----------------------
# Main matching function (returns 6-tuple per match)
# -----------------------
def find_object_matches(sf_data: pd.DataFrame, sole_data: pd.DataFrame) -> List[Tuple]:
    """
    Find potential object matches between SF and SOLE schemas.
    Returns list of tuples:
      (sf_name, sf_info, sole_name, sole_info, combined_score, overlap_data)
    """
    # Build object dicts from dataframes (auto-detect columns inside helper)
    sf_objects = build_objects_from_df(sf_data)
    sole_objects = build_objects_from_df(sole_data)

    print(f"📊 Found {len(sf_objects)} SF objects and {len(sole_objects)} SOLE objects")

    matches: List[Tuple] = []

    # For each SF object try to find the best SOLE candidate
    for sf_name, sf_info in sf_objects.items():
        best_match = None
        best_score = 0.0

        for sole_name, sole_info in sole_objects.items():
            # name similarity (0..1)
            name_similarity = calculate_name_similarity(sf_name, sole_name)

            # type compatibility (True/False)
            type_compatible = check_type_compatibility(sf_info.get('type', ''), sole_info.get('type', ''))

            # require some minimal name similarity OR exact name equality
            if not type_compatible:
                continue

            # compute overlap_data (dict)
            overlap_data = calculate_object_column_overlap(sf_info, sole_info)

            # combine name similarity and overlap_ratio into score
            combined_score = (name_similarity * 0.4 + overlap_data['overlap_ratio'] * 0.6) * 100.0

            # prefer exact name match strongly
            if sf_name == sole_name:
                combined_score = max(combined_score, 95.0)

            if combined_score > best_score:
                best_score = combined_score
                best_match = (sf_name, sf_info, sole_name, sole_info, round(combined_score, 2), overlap_data)

        # require minimum combined threshold to be considered a match
        if best_match and best_score > 50.0:
            matches.append(best_match)

    # sort matches descending by score
    matches.sort(key=lambda x: x[4], reverse=True)
    print(f"✅ Found {len(matches)} matches (score > 50).")
    return matches


# -----------------------
# Existing AI + rule-based analysis helpers (kept mostly unchanged)
# -----------------------
def analyze_schema_with_ai(sf_object: str, sf_type: str, sf_columns: List[str],
                           sole_object: str, sole_type: str, sole_columns: List[str],
                           overlap_data: Dict[str, Any], use_ai: bool = True) -> str:
    """
    Analyze schema migration with optional AI or fallback to rule-based analysis.
    (Same behavior as original.)
    """
    if use_ai:
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.2:3b',
                    'prompt': f"""Analyze this schema migration mapping between SF and SOLE:

SF Object: {sf_object} (Type: {sf_type})
SF Columns: {', '.join(sf_columns)}

SOLE Object: {sole_object} (Type: {sole_type})  
SOLE Columns: {', '.join(sole_columns)}

Column Analysis:
- Common columns: {overlap_data.get('common_columns', 0)}
- SF-only columns: {overlap_data.get('sf_only_columns', 0)}
- SOLE-only columns: {overlap_data.get('sole_only_columns', 0)}
- SF coverage ratio: {overlap_data.get('sf_coverage_ratio', 0):.2%}
- SOLE coverage ratio: {overlap_data.get('sole_coverage_ratio', 0):.2%}
- Matching strategy: {overlap_data.get('matching_strategy', 'unknown')}

Provide a concise migration assessment (2-3 sentences) covering:
1. Migration complexity (direct copy, column mapping, or complex restructure)
2. Key considerations or potential issues
3. Confidence level (high/medium/low)""",
                    'stream': False
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json().get('response', 'AI analysis unavailable')
        except Exception:
            pass  # fall back to rule-based

    return generate_rule_based_analysis(sf_object, sf_type, sf_columns, sole_object, sole_type, sole_columns,
                                        overlap_data)


def generate_rule_based_analysis(sf_object: str, sf_type: str, sf_columns: List[str],
                                 sole_object: str, sole_type: str, sole_columns: List[str],
                                 overlap_data: Dict[str, Any]) -> str:
    overlap_ratio = overlap_data.get('overlap_ratio', 0)
    common_cols = overlap_data.get('common_columns', 0)
    sf_only = overlap_data.get('sf_only_columns', 0)
    sole_only = overlap_data.get('sole_only_columns', 0)
    strategy = overlap_data.get('matching_strategy', 'unknown')

    if overlap_ratio >= 0.9:
        analysis = f"Excellent match with {overlap_ratio:.1%} column overlap. This object can be migrated with minimal transformation. "
    elif overlap_ratio >= 0.7:
        analysis = f"Good match with {overlap_ratio:.1%} column overlap requiring column mapping. {common_cols} common columns identified. "
    else:
        analysis = f"Complex migration with {overlap_ratio:.1%} column overlap. Significant restructuring required. "

    if strategy == "sf_focused_ignore_sole_extras":
        analysis += f"SOLE has {sole_only} extra columns that can be safely ignored during migration. "
    elif strategy == "sole_focused_ignore_sf_extras":
        analysis += f"SF has {sf_only} extra columns requiring careful data mapping strategy. "
    elif strategy == "standard_equal_columns":
        analysis += "Both objects have equal column counts enabling straightforward migration. "

    if sf_type in ['BASE TABLE', 'TABLE'] and sole_type in ['BASE TABLE', 'TABLE', 'MANAGED']:
        analysis += "Compatible table-to-managed migration with standard ETL processes."
    elif sf_type == 'VIEW' and sole_type == 'VIEW':
        analysis += "View-to-view migration may require definition adjustments."
    else:
        analysis += f"Cross-type migration ({sf_type} to {sole_type}) requires careful consideration."

    return analysis


# -----------------------
# Reporting helpers (kept unchanged)
# -----------------------
def generate_excel_report(results: Dict[str, Any], product_name: str, output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{product_name}_Schema_Migration_Analysis_{timestamp}.xlsx")
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        summary_data = {
            'Metric': [
                'SF Valid Tables', 'SF Valid Views', 'SOLE Total Tables', 'SOLE Total Views',
                'SOLE Managed Objects', 'Total Object Matches Found', 'Objects Coverage Percentage',
                'Average Object Similarity Score', 'High Confidence Mappings', 'Direct Copy Possible',
                'Column Mapping Required', 'Complex Restructures'
            ],
            'Value': [
                results['summary']['sf_tables_count'], results['summary']['sf_views_count'],
                results['summary']['sole_tables'], results['summary']['sole_views'],
                results['summary']['sole_managed'], results['summary']['total_matches'],
                f"{results['summary']['coverage_percentage']:.1f}%",
                f"{results['summary']['average_similarity']:.1f}%",
                results['summary']['high_confidence_matches'],
                results['summary']['direct_copy_objects'],
                results['summary']['column_mapping_objects'],
                results['summary']['complex_restructure_objects']
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Summary', index=False)
        if results['object_results']:
            pd.DataFrame(results['object_results']).to_excel(writer, sheet_name='Object Mapping', index=False)
    return filename


def generate_html_report(results: Dict[str, Any], product_name: str, output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{product_name}_Schema_Migration_Report_{timestamp}.html")
    coverage = results['summary']['coverage_percentage']
    avg_similarity = results['summary']['average_similarity']
    readiness_score = (coverage / 100) * (avg_similarity / 100) * 100
    if readiness_score >= 80:
        readiness_level = "🟢 EXCELLENT";
        readiness_color = "#28a745"
    elif readiness_score >= 70:
        readiness_level = "🟡 GOOD";
        readiness_color = "#ffc107"
    elif readiness_score >= 60:
        readiness_level = "🟠 MODERATE";
        readiness_color = "#fd7e14"
    else:
        readiness_level = "🔴 NEEDS WORK";
        readiness_color = "#dc3545"
    html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{product_name} Schema Migration Analysis</title></head><body>
    <h1>{product_name} Schema Migration Analysis</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>Migration Readiness Score: {readiness_score:.1f}% - {readiness_level}</h2>
    <p>Coverage: {coverage:.1f}% | Average Similarity: {avg_similarity:.1f}%</p>
    </body></html>"""
    with open(filename, 'w') as f:
        f.write(html_content)
    return filename


def generate_markdown_report(results: Dict[str, Any], product_name: str, output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{product_name}_Schema_Migration_Summary_{timestamp}.md")
    coverage = results['summary']['coverage_percentage']
    avg_similarity = results['summary']['average_similarity']
    readiness_score = (coverage / 100) * (avg_similarity / 100) * 100
    md = f"# {product_name} Schema Migration Analysis\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += f"**Migration Readiness Score:** {readiness_score:.1f}%\n\n"
    with open(filename, 'w') as f:
        f.write(md)
    return filename


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='Generic Schema Migration Analysis Tool')
    parser.add_argument('product_name', help='Name of the product being analyzed')
    parser.add_argument('sf_file', help='Path to SF schema file (CSV or Excel)')
    parser.add_argument('sole_file', help='Path to SOLE schema file (CSV or Excel)')
    parser.add_argument('--output-dir', default='.', help='Output directory for reports')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis (use rule-based analysis only)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    use_ai = not args.no_ai
    ai_status = "ENABLED" if use_ai else "DISABLED (--no-ai flag)"

    # quick check for Ollama (optional)
    if use_ai:
        try:
            resp = requests.get('http://localhost:11434/api/version', timeout=5)
            if resp.status_code == 200:
                ai_status = "ENABLED (Ollama connected)"
            else:
                ai_status = "FALLBACK (Ollama not responding)"
                use_ai = False
        except Exception:
            ai_status = "FALLBACK (Ollama not available)"
            use_ai = False

    print(f"\n🚀 Starting {args.product_name} Schema Migration Analysis")
    print("=" * 60)
    print(f"🤖 AI Analysis: {ai_status}")

    # Load schema data
    sf_data, sole_data = load_generic_schema_data(args.sf_file, args.sole_file)

    print("\n🔍 Finding object matches...")
    matches = find_object_matches(sf_data, sole_data)

    print(f"\n🔬 Analyzing matches with {'AI-powered' if use_ai else 'Rule-based'} analysis...")
    object_results = []

    # matches are tuples of length 6: (sf_name, sf_info, sole_name, sole_info, score, overlap_data)
    for sf_name, sf_info, sole_name, sole_info, score, overlap_data in matches:
        analysis = analyze_schema_with_ai(
            sf_name, sf_info.get('type', ''), sf_info.get('columns', []),
            sole_name, sole_info.get('type', ''), sole_info.get('columns', []),
            overlap_data, use_ai
        )

        if overlap_data['overlap_ratio'] >= 0.9:
            migration_complexity = 'direct_copy';
            confidence = 'high'
        elif overlap_data['overlap_ratio'] >= 0.7:
            migration_complexity = 'column_mapping';
            confidence = 'high' if overlap_data['overlap_ratio'] >= 0.8 else 'medium'
        else:
            migration_complexity = 'complex_restructure';
            confidence = 'low'

        object_results.append({
            'sf_object_name': sf_name,
            'sf_object_type': sf_info.get('type', ''),
            'sole_object_name': sole_name,
            'sole_object_type': sole_info.get('type', ''),
            'similarity_score': score,
            'migration_complexity': migration_complexity,
            'confidence': confidence,
            'ai_analysis': analysis,
            'sf_coverage_ratio': overlap_data.get('sf_coverage_ratio', 0),
            'sole_coverage_ratio': overlap_data.get('sole_coverage_ratio', 0),
            'matching_strategy': overlap_data.get('matching_strategy', ''),
            'common_columns': overlap_data.get('common_columns', 0),
            'sf_only_columns': overlap_data.get('sf_only_columns', 0),
            'sole_only_columns': overlap_data.get('sole_only_columns', 0)
        })

    # Build object dicts again for summary counts (consistent logic)
    sf_objects = build_objects_from_df(sf_data)
    sole_objects = build_objects_from_df(sole_data)

    sf_tables_count = len([t for t in sf_objects.values() if t.get('type', '') in ['BASE TABLE', 'TABLE']])
    sf_views_count = len([t for t in sf_objects.values() if t.get('type', '') == 'VIEW'])
    sole_tables = len([t for t in sole_objects.values() if t.get('type', '') in ['BASE TABLE', 'TABLE']])
    sole_views = len([t for t in sole_objects.values() if t.get('type', '') == 'VIEW'])
    sole_managed = len([t for t in sole_objects.values() if t.get('type', '') == 'MANAGED'])

    total_sf_objects = len(sf_objects)
    coverage = (len(object_results) / total_sf_objects * 100) if total_sf_objects > 0 else 0
    avg_similarity = (sum(r['similarity_score'] for r in object_results) / len(object_results)) if object_results else 0

    results = {
        'summary': {
            'sf_tables_count': sf_tables_count,
            'sf_views_count': sf_views_count,
            'sole_tables': sole_tables,
            'sole_views': sole_views,
            'sole_managed': sole_managed,
            'total_matches': len(object_results),
            'coverage_percentage': coverage,
            'average_similarity': avg_similarity,
            'high_confidence_matches': len([r for r in object_results if r['confidence'] == 'high']),
            'direct_copy_objects': len([r for r in object_results if r['migration_complexity'] == 'direct_copy']),
            'column_mapping_objects': len([r for r in object_results if r['migration_complexity'] == 'column_mapping']),
            'complex_restructure_objects': len(
                [r for r in object_results if r['migration_complexity'] == 'complex_restructure'])
        },
        'object_results': object_results
    }

    print("\n📊 Generating reports...")
    excel_file = generate_excel_report(results, args.product_name, args.output_dir)
    html_file = generate_html_report(results, args.product_name, args.output_dir)
    markdown_file = generate_markdown_report(results, args.product_name, args.output_dir)

    json_file = os.path.join(args.output_dir,
                             f"{args.product_name}_Schema_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n🎉 {args.product_name} Schema Migration Analysis Complete!")
    print("=" * 60)
    print(f"📊 Analyzed {len(object_results)} object mappings")
    print(f"🎯 Migration Readiness Score: {(coverage / 100) * (avg_similarity / 100) * 100:.1f}%")
    print(f"📈 Schema Coverage: {coverage:.1f}%")
    print(f"📈 Average Similarity: {avg_similarity:.1f}%")
    print(f"⚡ Direct Copy Objects: {results['summary']['direct_copy_objects']}")
    print(f"🔗 Column Mapping Objects: {results['summary']['column_mapping_objects']}")
    print(f"🔧 Complex Restructures: {results['summary']['complex_restructure_objects']}")
    print(
        f"\n📁 Generated Files:\n  📊 Excel Report: {excel_file}\n  🌐 HTML Report: {html_file}\n  📝 Markdown Summary: {markdown_file}\n  📄 JSON Data: {json_file}")


if __name__ == "__main__":
    main()
