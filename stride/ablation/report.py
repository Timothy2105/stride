"""Ablation report generator for STRIDE.

Generates a premium, interactive HTML dashboard to visualize ablation results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def generate_html_report(
    results_dir: str = "ablations/results",
    plots_dir: str = "ablations/plots",
    out_path: str = "ablations/index.html",
):
    """Generate a premium HTML dashboard for ablation results."""
    
    # 1. Gather all plots
    plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")])
    
    # Identify unique groups/factors based on plot filenames
    # Filenames are like: component_ablation.png, edit_scale_sweep.png, etc.
    factors = []
    for f in plot_files:
        name = f.replace(".png", "").replace("_", " ").title()
        factors.append({
            "id": f.replace(".png", ""),
            "name": name,
            "filename": f
        })

    # 2. Build Sidebar and Content sections
    sidebar_items = ""
    content_sections = ""
    
    for factor in factors:
        sidebar_items += f'<a href="#{factor["id"]}" class="sidebar-item" onclick="showSection(\'{factor["id"]}\')">{factor["name"]}</a>\n'
        
        content_sections += f'''
        <section id="{factor["id"]}" class="tab-content">
            <div class="card">
                <h2>{factor["name"]}</h2>
                <div class="plot-container">
                    <img src="plots/{factor["filename"]}" alt="{factor["name"]}">
                </div>
            </div>
        </section>
        '''

    # 3. Final HTML Template
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STRIDE Ablated Ecosystem</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0d1117;
            --sidebar-bg: #161b22;
            --accent-color: #58a6ff;
            --text-main: #c9d1d9;
            --text-dim: #8b949e;
            --card-bg: #161b22;
            --hover-bg: #21262d;
            --border-color: #30363d;
            --sidebar-width: 280px;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}

        /* Sidebar Styles */
        .sidebar {{
            width: var(--sidebar-width);
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            padding: 2rem 1.5rem;
        }}

        .logo {{
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 1.8rem;
            color: white;
            margin-bottom: 3rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }}

        .logo-icon {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #58a6ff, #1f6feb);
            border-radius: 8px;
        }}

        .sidebar-item {{
            text-decoration: none;
            color: var(--text-dim);
            padding: 0.8rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            transition: all 0.2s ease;
            font-weight: 500;
            cursor: pointer;
        }}

        .sidebar-item:hover {{
            background-color: var(--hover-bg);
            color: var(--text-main);
        }}

        .sidebar-item.active {{
            background-color: rgba(88, 166, 255, 0.1);
            color: var(--accent-color);
            border-left: 3px solid var(--accent-color);
        }}

        /* Main Content Styles */
        .main-content {{
            flex: 1;
            padding: 3rem;
            overflow-y: auto;
            position: relative;
        }}

        .header {{
            margin-bottom: 2rem;
        }}

        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: white;
            letter-spacing: -0.02em;
        }}

        .subtitle {{
            color: var(--text-dim);
            margin-top: 0.5rem;
            font-size: 1.1rem;
        }}

        /* Card Styles */
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        h2 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            margin-bottom: 1.5rem;
            color: var(--accent-color);
        }}

        .plot-container {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff05;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .plot-container img {{
            max-width: 100%;
            border-radius: 4px;
            transition: transform 0.3s ease;
        }}

        .plot-container img:hover {{
            transform: scale(1.02);
        }}

        /* Tab Logic */
        .tab-content {{
            display: none;
            animation: fadeIn 0.4s ease;
        }}

        .active-tab {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: var(--bg-color);
        }}
        ::-webkit-scrollbar-thumb {{
            background: var(--border-color);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-dim);
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="logo">
            <div class="logo-icon"></div>
            STRIDE
        </div>
        <div class="sidebar-label" style="text-transform: uppercase; font-size: 0.7rem; color: var(--text-dim); letter-spacing: 0.1em; margin-bottom: 1rem; padding-left: 1rem;">Ablation Factors</div>
        {sidebar_items}
    </div>

    <div class="main-content">
        <div class="header">
            <h1>Ablation Dashboard</h1>
            <p class="subtitle">Comprehensive analysis of the STRIDE architecture and hyper-parameters.</p>
        </div>

        {content_sections}
        
        <div id="welcome" class="tab-content active-tab" style="max-width: 800px;">
            <div class="card">
                <h2>Welcome to STRIDE Ablations</h2>
                <p style="line-height: 1.6; color: var(--text-dim);">
                    Select a factor from the left sidebar to view the corresponding ablation charts. 
                    These experiments explore the sensitivity of the STRIDE pipeline to various 
                    component omissions and hyper-parameter adjustments.
                </p>
                <div style="margin-top: 1.5rem; display: flex; gap: 1rem;">
                    <div style="background: rgba(88, 166, 255, 0.1); padding: 1rem; border-radius: 8px; flex: 1; border: 1px solid rgba(88, 166, 255, 0.2);">
                        <div style="color: var(--accent-color); font-weight: 600; margin-bottom: 0.5rem;">Results Dir</div>
                        <code>{results_dir}</code>
                    </div>
                    <div style="background: rgba(88, 166, 255, 0.1); padding: 1rem; border-radius: 8px; flex: 1; border: 1px solid rgba(88, 166, 255, 0.2);">
                        <div style="color: var(--accent-color); font-weight: 600; margin-bottom: 0.5rem;">Plots Dir</div>
                        <code>{plots_dir}</code>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function showSection(sectionId) {{
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active-tab'));
            
            // Show selected tab
            const target = document.getElementById(sectionId);
            if (target) target.classList.add('active-tab');
            
            // Update sidebar active state
            const items = document.querySelectorAll('.sidebar-item');
            items.forEach(item => {{
                item.classList.remove('active');
                if (item.getAttribute('onclick').includes(sectionId)) {{
                    item.classList.add('active');
                }}
            }});
        }}
    </script>
</body>
</html>
'''

    # Write to file
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html_content)
    
    print(f"Generated dashboard: {out_path}")


if __name__ == "__main__":
    generate_html_report()
