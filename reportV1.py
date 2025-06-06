# report.py
from fpdf import FPDF
from datetime import datetime

def generate_pdf_report(filename, params, results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Fusion Reactor Feasibility Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Input Parameters", ln=True)
    pdf.set_font("Arial", '', 10)
    for key, value in params.items():
        pdf.cell(95, 8, f"{key.replace('_', ' ').title()}:", border=1)
        pdf.cell(95, 8, f"{value:.3e}", border=1, ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Simulation Results", ln=True)
    pdf.set_font("Arial", '', 10)
    for key, value in results.items():
        pdf.cell(95, 8, f"{key.replace('_', ' ').title()}:", border=1)
        pdf.cell(95, 8, f"{value:.4e}", border=1, ln=True)
    pdf.output(filename)

def generate_html_report(filename, params, results):
    html = f"""
    <html><head><title>Fusion Reactor Feasibility Report</title><style>
    body {{ font-family: sans-serif; margin: 2em; }} h1, h2 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 80%; margin-top: 1em; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ background-color: #f2f2f2; }} </style></head>
    <body><h1>Fusion Reactor Feasibility Report</h1>
    <p>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>Input Parameters</h2><table><tr><th>Parameter</th><th>Value</th></tr>
    """
    for key, value in params.items(): html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.3e}</td></tr>"
    html += "</table><h2>Simulation Results</h2><table><tr><th>Result</th><th>Value</th></tr>"
    for key, value in results.items(): html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.4e}</td></tr>"
    html += "</table></body></html>"
    with open(filename, 'w') as f: f.write(html)
