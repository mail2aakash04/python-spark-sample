#!/usr/bin/env python3
"""
Generic Schema Migration Analysis Tool
Powered by Ollama Llama 3.2 3B for AI-driven schema comparison and migration strategy

Usage: python3 generic_schema_analysis.py <product_name> <sf_file> <sole_file> [--output-dir OUTPUT_DIR]
"""

import pandas as pd
import requests
import json
import re
from datetime import datetime
import argparse
import os
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


def load_generic_schema_data(sf_file: str, sole_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generic function to load schema data from SF and SOLE files.
    Supports both CSV and Excel formats.
    """
    print(f"📊 Loading SF schema from: {sf_file}")
    print(f"📊 Loading SOLE schema from: {sole_file}")

    # Load SF data
    if sf_file.endswith('.xlsx'):
        sf_data = pd.read_excel(sf_file)
    else:
        sf_data = pd.read_csv(sf_file)

    # Load SOLE data
    if sole_file.endswith('.xlsx'):
        sole_data = pd.read_excel(sole_file)
    else:
        sole_data = pd.read_csv(sole_file)

    print(f"✅ SF data loaded: {len(sf_data)} rows")
    print(f"✅ SOLE data loaded: {len(sole_data)} rows")

    return sf_data, sole_data


def analyze_schema_with_ai(sf_object: str, sf_type: str, sf_columns: List[str],
                           sole_object: str, sole_type: str, sole_columns: List[str],
                           overlap_data: Dict[str, Any], use_ai: bool = True) -> str:
    """
    Analyze schema migration with optional AI or fallback to rule-based analysis.
    """
    # Try AI analysis first if enabled and available
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
            pass  # Fall through to rule-based analysis

    # Fallback to rule-based analysis
    return generate_rule_based_analysis(sf_object, sf_type, sf_columns, sole_object, sole_type, sole_columns,
                                        overlap_data)


def generate_rule_based_analysis(sf_object: str, sf_type: str, sf_columns: List[str],
                                 sole_object: str, sole_type: str, sole_columns: List[str],
                                 overlap_data: Dict[str, Any]) -> str:
    """
    Generate rule-based migration analysis without AI dependency.
    """
    overlap_ratio = overlap_data.get('overlap_ratio', 0)
    common_cols = overlap_data.get('common_columns', 0)
    sf_only = overlap_data.get('sf_only_columns', 0)
    sole_only = overlap_data.get('sole_only_columns', 0)
    strategy = overlap_data.get('matching_strategy', 'unknown')

    # Determine complexity based on overlap ratio
    if overlap_ratio >= 0.9:
        complexity = "direct copy"
        confidence = "high"
        analysis = f"Excellent match with {overlap_ratio:.1%} column overlap. This object can be migrated with minimal transformation. "
    elif overlap_ratio >= 0.7:
        complexity = "column mapping"
        confidence = "medium" if overlap_ratio >= 0.8 else "medium"
        analysis = f"Good match with {overlap_ratio:.1%} column overlap requiring column mapping. {common_cols} common columns identified. "
    else:
        complexity = "complex restructure"
        confidence = "low"
        analysis = f"Complex migration with {overlap_ratio:.1%} column overlap. Significant restructuring required. "

    # Add strategy-specific insights
    if strategy == "sf_focused_ignore_sole_extras":
        analysis += f"SOLE has {sole_only} extra columns that can be safely ignored during migration. "
    elif strategy == "sole_focused_ignore_sf_extras":
        analysis += f"SF has {sf_only} extra columns requiring careful data mapping strategy. "
    elif strategy == "standard_equal_columns":
        analysis += "Both objects have equal column counts enabling straightforward migration. "

    # Add type-specific recommendations
    if sf_type in ['BASE TABLE', 'TABLE'] and sole_type in ['BASE TABLE', 'TABLE', 'MANAGED']:
        analysis += "Compatible table-to-managed migration with standard ETL processes."
    elif sf_type == 'VIEW' and sole_type == 'VIEW':
        analysis += "View-to-view migration may require definition adjustments."
    else:
        analysis += f"Cross-type migration ({sf_type} to {sole_type}) requires careful consideration."

    return analysis


def calculate_object_column_overlap(sf_object_info: Dict, sole_object_info: Dict) -> Dict[str, Any]:
    """
    Calculate detailed column overlap between two objects with enhanced matching logic.
    """
    sf_cols = set(col.upper() for col in sf_object_info.get('columns', []))
    sole_cols = set(col.upper() for col in sole_object_info.get('columns', []))

    common_cols = sf_cols.intersection(sole_cols)
    sf_only = sf_cols - sole_cols
    sole_only = sole_cols - sf_cols

    sf_column_count = len(sf_cols)
    sole_column_count = len(sole_cols)

    # Calculate different overlap ratios
    total_unique_cols = len(sf_cols.union(sole_cols))
    standard_overlap_ratio = len(common_cols) / total_unique_cols if total_unique_cols > 0 else 0

    sf_coverage_ratio = len(common_cols) / sf_column_count if sf_column_count > 0 else 0
    sole_coverage_ratio = len(common_cols) / sole_column_count if sole_column_count > 0 else 0

    # Determine matching strategy
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
        'match_quality': match_quality
    }


from typing import List, Tuple


def find_object_matches(sf_data: pd.DataFrame, sole_data: pd.DataFrame) -> List[Tuple]:
    """
    Find potential object matches between SF and SOLE schemas.
    Enhanced to handle SOLE MANAGED = SF TABLES mapping.
    """
    matches = []

    # Group SF data by object
    sf_objects = {}
    for _, row in sf_data.iterrows():
        obj_name = str(row.get('TABLE_NAME', row.get('Object_Name', ''))).strip().upper()
        obj_type = str(row.get('TABLE_TYPE', row.get('Object_Type', ''))).strip().upper()
        column_name = str(row.get('COLUMN_NAME', row.get('Column_Name', ''))).strip().upper()

        if obj_name and obj_name != 'NAN':
            if obj_name not in sf_objects:
                sf_objects[obj_name] = {'type': obj_type, 'columns': []}
            if column_name and column_name != 'NAN':
                sf_objects[obj_name]['columns'].append(column_name)

    # Group SOLE data by object
    sole_objects = {}
    for _, row in sole_data.iterrows():
        obj_name = str(row.get('TABLE_NAME', row.get('Object_Name', ''))).strip().upper()
        obj_type = str(row.get('TABLE_TYPE', row.get('Object_Type', ''))).strip().upper()
        column_name = str(row.get('COLUMN_NAME', row.get('Column_Name', ''))).strip().upper()

        if obj_name and obj_name != 'NAN':
            if obj_name not in sole_objects:
                sole_objects[obj_name] = {'type': obj_type, 'columns': []}
            if column_name and column_name != 'NAN':
                sole_objects[obj_name]['columns'].append(column_name)

    print(f"📊 Found {len(sf_objects)} SF objects and {len(sole_objects)} SOLE objects")

    # Find matches using fuzzy name matching and type compatibility
    for sf_name, sf_info in sf_objects.items():
        best_match = None
        best_score = 0

        for sole_name, sole_info in sole_objects.items():
            # Calculate name similarity
            name_similarity = calculate_name_similarity(sf_name, sole_name)

            # Type compatibility check (SOLE MANAGED = SF TABLE)
            type_compatible = check_type_compatibility(sf_info['type'], sole_info['type'])

            if type_compatible and name_similarity > 0.6:  # Threshold for consideration
                # Calculate column overlap
                overlap_data = calculate_object_column_overlap(sf_info, sole_info)

                # Combined score: name similarity + column overlap
                combined_score = (name_similarity * 0.4 + overlap_data['overlap_ratio'] * 0.6) * 100

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = (sf_name, sf_info, sole_name, sole_info, combined_score, overlap_data)

        if best_match and best_score > 50:  # Minimum threshold
            matches.append(best_match)

    # Sort by score (highest first)
    matches.sort(key=lambda x: x[4], reverse=True)
    return matches


def calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two object names."""
    name1 = name1.upper().strip()
    name2 = name2.upper().strip()

    if name1 == name2:
        return 1.0

    # Simple similarity based on common characters
    set1 = set(name1)
    set2 = set(name2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    return len(intersection) / len(union) if union else 0


def check_type_compatibility(sf_type: str, sole_type: str) -> bool:
    """Check if SF and SOLE object types are compatible for migration."""
    sf_type = sf_type.upper().strip()
    sole_type = sole_type.upper().strip()

    # Direct matches
    if sf_type == sole_type:
        return True

    # Enhanced compatibility: SOLE MANAGED = SF TABLE
    if sf_type in ['BASE TABLE', 'TABLE'] and sole_type in ['BASE TABLE', 'TABLE', 'MANAGED']:
        return True

    # Views should match views
    if sf_type == 'VIEW' and sole_type == 'VIEW':
        return True

    return False


def generate_excel_report(results: Dict[str, Any], product_name: str, output_dir: str) -> str:
    """Generate comprehensive Excel report with multiple tabs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{product_name}_Schema_Migration_Analysis_{timestamp}.xlsx")

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Executive Summary
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
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)

        # Object Mapping Details
        if results['object_results']:
            object_df = pd.DataFrame(results['object_results'])
            object_df.to_excel(writer, sheet_name='Object Mapping', index=False)

        # Migration Strategy
        strategy_data = {
            'Phase': ['Phase 1', 'Phase 2', 'Phase 3'],
            'Focus': ['Direct Copy Objects', 'Column Mapping Objects', 'Complex Restructures'],
            'Description': [
                f"Migrate {results['summary']['direct_copy_objects']} objects with direct schema compatibility",
                f"Migrate {results['summary']['column_mapping_objects']} objects requiring column mapping",
                f"Restructure {results['summary']['complex_restructure_objects']} objects with significant differences"
            ]
        }
        strategy_df = pd.DataFrame(strategy_data)
        strategy_df.to_excel(writer, sheet_name='Migration Strategy', index=False)

        # AI Analysis Details
        ai_data = []
        for result in results['object_results']:
            ai_data.append({
                'SF_Object': result.get('sf_object_name', ''),
                'SF_Type': result.get('sf_object_type', ''),
                'SOLE_Object': result.get('sole_object_name', ''),
                'SOLE_Type': result.get('sole_object_type', ''),
                'AI_Analysis': result.get('ai_analysis', ''),
                'Similarity_Score': result.get('similarity_score', 0),
                'Migration_Complexity': result.get('migration_complexity', ''),
                'Confidence_Level': result.get('confidence', '')
            })

        if ai_data:
            ai_df = pd.DataFrame(ai_data)
            ai_df.to_excel(writer, sheet_name='AI Analysis', index=False)

    return filename


def generate_html_report(results: Dict[str, Any], product_name: str, output_dir: str) -> str:
    """Generate interactive HTML report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{product_name}_Schema_Migration_Report_{timestamp}.html")

    # Calculate migration readiness score
    coverage = results['summary']['coverage_percentage']
    avg_similarity = results['summary']['average_similarity']
    readiness_score = (coverage / 100) * (avg_similarity / 100) * 100

    # Determine readiness level
    if readiness_score >= 80:
        readiness_level = "🟢 EXCELLENT"
        readiness_color = "#28a745"
    elif readiness_score >= 70:
        readiness_level = "🟡 GOOD"
        readiness_color = "#ffc107"
    elif readiness_score >= 60:
        readiness_level = "🟠 MODERATE"
        readiness_color = "#fd7e14"
    else:
        readiness_level = "🔴 NEEDS WORK"
        readiness_color = "#dc3545"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{product_name} Schema Migration Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
            .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
            .readiness {{ background: {readiness_color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0; }}
            .section {{ margin: 30px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 {product_name} Schema Migration Analysis</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="readiness">
            <h2>Migration Readiness Score: {readiness_score:.1f}%</h2>
            <h3>{readiness_level}</h3>
        </div>

        <div class="summary">
            <div class="metric">
                <h3>📊 Schema Coverage</h3>
                <p><strong>{coverage:.1f}%</strong> of objects matched</p>
            </div>
            <div class="metric">
                <h3>🎯 Average Similarity</h3>
                <p><strong>{avg_similarity:.1f}%</strong> schema alignment</p>
            </div>
            <div class="metric">
                <h3>⚡ Direct Copy Objects</h3>
                <p><strong>{results['summary']['direct_copy_objects']}</strong> ready for migration</p>
            </div>
            <div class="metric">
                <h3>🔗 Column Mapping Required</h3>
                <p><strong>{results['summary']['column_mapping_objects']}</strong> objects need mapping</p>
            </div>
        </div>

        <div class="section">
            <h2>📈 Migration Strategy</h2>
            <p><strong>Phase 1:</strong> Migrate {results['summary']['direct_copy_objects']} direct copy objects</p>
            <p><strong>Phase 2:</strong> Handle {results['summary']['column_mapping_objects']} objects requiring column mapping</p>
            <p><strong>Phase 3:</strong> Restructure {results['summary']['complex_restructure_objects']} complex objects</p>
        </div>
    </body>
    </html>
    """

    with open(filename, 'w') as f:
        f.write(html_content)

    return filename


def generate_markdown_report(results: Dict[str, Any], product_name: str, output_dir: str) -> str:
    """Generate Markdown documentation report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{product_name}_Schema_Migration_Summary_{timestamp}.md")

    coverage = results['summary']['coverage_percentage']
    avg_similarity = results['summary']['average_similarity']
    readiness_score = (coverage / 100) * (avg_similarity / 100) * 100

    markdown_content = f"""# {product_name} Schema Migration Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Migration Readiness Score: {readiness_score:.1f}%

### Executive Summary

- **Schema Coverage:** {coverage:.1f}% of SF objects have SOLE matches
- **Average Similarity:** {avg_similarity:.1f}% schema alignment
- **Total Matches Found:** {results['summary']['total_matches']} object mappings
- **High Confidence Matches:** {results['summary']['high_confidence_matches']} objects

### Migration Complexity Breakdown

| Complexity Level | Object Count | Description |
|------------------|--------------|-------------|
| Direct Copy | {results['summary']['direct_copy_objects']} | Objects ready for direct migration |
| Column Mapping | {results['summary']['column_mapping_objects']} | Objects requiring column transformation |
| Complex Restructure | {results['summary']['complex_restructure_objects']} | Objects needing significant changes |

### Migration Strategy

1. **Phase 1 - Direct Migration:** Start with {results['summary']['direct_copy_objects']} objects that can be copied directly
2. **Phase 2 - Column Mapping:** Handle {results['summary']['column_mapping_objects']} objects requiring field mapping
3. **Phase 3 - Restructuring:** Address {results['summary']['complex_restructure_objects']} objects with complex differences

### Recommendations

- Focus on high-confidence matches first to establish migration patterns
- Develop column mapping templates for phase 2 objects
- Plan additional analysis for complex restructure objects

---
*Report generated by Generic Schema Migration Analysis Tool*
"""

    with open(filename, 'w') as f:
        f.write(markdown_content)

    return filename


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Generic Schema Migration Analysis Tool')
    parser.add_argument('product_name', help='Name of the product being analyzed')
    parser.add_argument('sf_file', help='Path to SF schema file (CSV or Excel)')
    parser.add_argument('sole_file', help='Path to SOLE schema file (CSV or Excel)')
    parser.add_argument('--output-dir', default='.', help='Output directory for reports')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis (use rule-based analysis only)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check AI availability
    use_ai = not args.no_ai
    ai_status = "DISABLED (--no-ai flag)" if args.no_ai else "ENABLED"

    # Test AI connection if enabled
    if use_ai:
        try:
            response = requests.get('http://localhost:11434/api/version', timeout=5)
            if response.status_code == 200:
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

    # Find object matches
    print("\n🔍 Finding object matches...")
    matches = find_object_matches(sf_data, sole_data)

    # Process matches with AI or rule-based analysis
    analysis_type = "AI-powered" if use_ai else "Rule-based"
    print(f"\n🔬 Analyzing matches with {analysis_type} analysis...")
    object_results = []

    for sf_name, sf_info, sole_name, sole_info, score, overlap_data in matches:
        # Get analysis (AI or rule-based)
        analysis = analyze_schema_with_ai(
            sf_name, sf_info['type'], sf_info['columns'],
            sole_name, sole_info['type'], sole_info['columns'],
            overlap_data, use_ai
        )

        # Determine migration complexity
        if overlap_data['overlap_ratio'] >= 0.9:
            migration_complexity = 'direct_copy'
            confidence = 'high'
        elif overlap_data['overlap_ratio'] >= 0.7:
            migration_complexity = 'column_mapping'
            confidence = 'high' if overlap_data['overlap_ratio'] >= 0.8 else 'medium'
        else:
            migration_complexity = 'complex_restructure'
            confidence = 'low'

        object_results.append({
            'sf_object_name': sf_name,
            'sf_object_type': sf_info['type'],
            'sole_object_name': sole_name,
            'sole_object_type': sole_info['type'],
            'similarity_score': score,
            'migration_complexity': migration_complexity,
            'confidence': confidence,
            'ai_analysis': analysis,
            'sf_coverage_ratio': overlap_data['sf_coverage_ratio'],
            'sole_coverage_ratio': overlap_data['sole_coverage_ratio'],
            'matching_strategy': overlap_data['matching_strategy'],
            'common_columns': overlap_data['common_columns'],
            'sf_only_columns': overlap_data['sf_only_columns'],
            'sole_only_columns': overlap_data['sole_only_columns']
        })

    # Calculate summary statistics
    sf_objects = {}
    sole_objects = {}

    # Count SF objects
    for _, row in sf_data.iterrows():
        obj_name = str(row.get('TABLE_NAME', row.get('Object_Name', ''))).strip().upper()
        obj_type = str(row.get('TABLE_TYPE', row.get('Object_Type', ''))).strip().upper()
        if obj_name and obj_name != 'NAN':
            sf_objects[obj_name] = obj_type

    # Count SOLE objects
    for _, row in sole_data.iterrows():
        obj_name = str(row.get('TABLE_NAME', row.get('Object_Name', ''))).strip().upper()
        obj_type = str(row.get('TABLE_TYPE', row.get('Object_Type', ''))).strip().upper()
        if obj_name and obj_name != 'NAN':
            sole_objects[obj_name] = obj_type

    sf_tables_count = len([t for t in sf_objects.values() if t in ['BASE TABLE', 'TABLE']])
    sf_views_count = len([t for t in sf_objects.values() if t == 'VIEW'])
    sole_tables = len([t for t in sole_objects.values() if t in ['BASE TABLE', 'TABLE']])
    sole_views = len([t for t in sole_objects.values() if t == 'VIEW'])
    sole_managed = len([t for t in sole_objects.values() if t == 'MANAGED'])

    total_sf_objects = len(sf_objects)
    coverage = len(object_results) / total_sf_objects * 100 if total_sf_objects > 0 else 0
    avg_similarity = sum(r['similarity_score'] for r in object_results) / len(object_results) if object_results else 0

    # Compile results
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

    # Generate reports
    print("\n📊 Generating reports...")

    excel_file = generate_excel_report(results, args.product_name, args.output_dir)
    html_file = generate_html_report(results, args.product_name, args.output_dir)
    markdown_file = generate_markdown_report(results, args.product_name, args.output_dir)

    # Save JSON data
    json_file = os.path.join(args.output_dir,
                             f"{args.product_name}_Schema_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print final summary
    print(f"\n🎉 {args.product_name} Schema Migration Analysis Complete!")
    print("=" * 60)
    print(f"📊 Analyzed {len(object_results)} object mappings")
    print(f"🎯 Migration Readiness Score: {(coverage / 100) * (avg_similarity / 100) * 100:.1f}%")
    print(f"📈 Schema Coverage: {coverage:.1f}%")
    print(f"📈 Average Similarity: {avg_similarity:.1f}%")
    print(f"⚡ Direct Copy Objects: {results['summary']['direct_copy_objects']}")
    print(f"🔗 Column Mapping Objects: {results['summary']['column_mapping_objects']}")
    print(f"🔧 Complex Restructures: {results['summary']['complex_restructure_objects']}")

    print(f"\n📁 Generated Files:")
    print(f"  📊 Excel Report: {excel_file}")
    print(f"  🌐 HTML Report: {html_file}")
    print(f"  📝 Markdown Summary: {markdown_file}")
    print(f"  📄 JSON Data: {json_file}")


if __name__ == "__main__":
    main()
