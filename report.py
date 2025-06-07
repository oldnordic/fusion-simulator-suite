# report.py
from fpdf import FPDF
from datetime import datetime
import os

OVOID_COIL_GEOMETRY_PARAMS_KEYS = [
    "geometry", "max_radius_coil_assembly", "total_height_coil_assembly",
    "num_spirals_set1", "pitch_set1", "num_turns_set1",
    "num_spirals_set2", "pitch_set2", "num_turns_set2",
    "points_per_coil", "total_current_MA_coil_assembly"
]

def format_key_for_display(key):
    return key.replace('_', ' ').title()

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 16)
        self.cell(0, 10, "Fusion Reactor Feasibility Report", ln=True, align='C')
        self.set_font("Arial", '', 10)
        self.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        # --- CORRECTED: Improved page break logic ---
        if self.get_y() > 240:
            self.add_page()
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def key_value_table(self, data):
        self.set_font("Arial", '', 10)
        col_width_key = 95
        col_width_val = self.w - self.l_margin - self.r_margin - col_width_key
        
        for key, value in data.items():
            # --- CORRECTED: Use MultiCell for values to allow text wrapping ---
            y_before = self.get_y()
            
            # Key cell
            self.cell(col_width_key, 8, f"{format_key_for_display(key)}:", border=1)
            
            # Value cell using MultiCell
            # The 'ln=3' parameter moves cursor to beginning of next line after both cells
            self.multi_cell(col_width_val, 8, str(value), border=1, ln=3) 
            
            # Ensure the next line starts correctly after a potential multi-line cell
            y_after = self.get_y()
            if y_after < y_before + 8: # If MultiCell was only one line
                 self.set_y(y_before + 8)

            if self.get_y() > 260: # Page break for long tables
                self.add_page()
        self.ln(5)

def generate_pdf_report(filename, params, results, image_path=None):
    pdf = PDF()
    pdf.add_page()

    general_params, coil_specific_params = {}, {}
    is_ovoid_design = params.get("geometry") == "ovoid_fibonacci_reactor"
    for key, value in params.items():
        if is_ovoid_design and key in OVOID_COIL_GEOMETRY_PARAMS_KEYS:
            coil_specific_params[key] = value
        elif key not in OVOID_COIL_GEOMETRY_PARAMS_KEYS:
            general_params[key] = value

    pdf.section_title("General Input Parameters")
    pdf.key_value_table(general_params)

    if coil_specific_params:
        pdf.section_title("Ovoid Fibonacci Coil Design Parameters")
        pdf.key_value_table(coil_specific_params)

    if image_path and os.path.exists(image_path):
        try:
            # --- CORRECTED: Better page break logic for image ---
            if pdf.get_y() > 150: 
                pdf.add_page()
            pdf.section_title("Reactor Coil Geometry Visualization")
            page_width = pdf.w - 2 * pdf.l_margin
            pdf.image(image_path, x=pdf.l_margin + (page_width - page_width*0.8)/2, w=page_width*0.8)
            pdf.ln(5)
        except Exception as e:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 8, f"(Could not embed image: {e})", ln=True, align='C')
            pdf.set_text_color(0, 0, 0)
    
    pdf.section_title("Performance & Stability Results")
    results_formatted = {k: (f"{v:.4g}" if isinstance(v, (int, float)) else v) for k, v in results.items()}
    pdf.key_value_table(results_formatted)

    pdf.output(filename)

def generate_html_report(filename, params, results, image_path=None):
    # --- CORRECTED: Implemented the HTML report generation function ---
    general_params, coil_specific_params = {}, {}
    is_ovoid_design = params.get("geometry") == "ovoid_fibonacci_reactor"
    for key, value in params.items():
        if is_ovoid_design and key in OVOID_COIL_GEOMETRY_PARAMS_KEYS:
            coil_specific_params[key] = value
        elif key not in OVOID_COIL_GEOMETRY_PARAMS_KEYS:
            general_params[key] = value

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fusion Reactor Feasibility Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 80%; margin: 20px auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .container {{ max-width: 1000px; margin: auto; }}
            .report-header {{ text-align: center; margin-bottom: 40px; }}
            .report-header p {{ color: #7f8c8d; font-size: 0.9em; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; border: 1px solid #ddd; padding: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="report-header">
                <h1>Fusion Reactor Feasibility Report</h1>
                <p>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <h2>General Input Parameters</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
    """
    for key, value in general_params.items():
        html += f"<tr><td>{format_key_for_display(key)}</td><td>{value}</td></tr>"
    html += "</table>"

    if coil_specific_params:
        html += "<h2>Ovoid Fibonacci Coil Design Parameters</h2><table><tr><th>Parameter</th><th>Value</th></tr>"
        for key, value in coil_specific_params.items():
            val_str = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
            html += f"<tr><td>{format_key_for_display(key)}</td><td>{val_str}</td></tr>"
        html += "</table>"

    if image_path and os.path.exists(image_path):
        html += f"""
            <h2>Reactor Coil Geometry Visualization</h2>
            <img src="{image_path}" alt="Reactor Coil Geometry">
        """

    html += "<h2>Performance & Stability Results</h2><table><tr><th>Parameter</th><th>Value</th></tr>"
    for key, value in results.items():
        val_str = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
        html += f"<tr><td>{format_key_for_display(key)}</td><td>{val_str}</td></tr>"
    html += "</table>"

    html += """
        </div>
    </body>
    </html>
    """
    with open(filename, 'w') as f:
        f.write(html)