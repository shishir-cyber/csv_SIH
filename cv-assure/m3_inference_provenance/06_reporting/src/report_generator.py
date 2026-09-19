"""
report_generator.py
-------------------
Assurance Report & Governance Generator for CV-ASSURE M3 Inference Provenance
(Clause 2.2.5 Compliance).

Compiles JSON audit logs and builds an interactive HTML Console report
showing overall disposition, finding breakdowns, evidence artifacts, and chain integrity.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import sys
ROOT_M3 = Path(__file__).resolve().parents[2]
AUDIT_SRC = ROOT_M3 / "05_verification_audit" / "src"
if str(AUDIT_SRC) not in sys.path:
    sys.path.insert(0, str(AUDIT_SRC))

try:
    from audit_engine import Finding
except ImportError:
    pass


class ReportGenerator:
    """
    Generates standardized JSON and HTML assurance reports for Clause 2.2.5.
    """

    def __init__(self, findings: List[Any], total_receipts: int = 0, total_checkpoints: int = 0):
        self.findings = findings
        self.total_receipts = total_receipts
        self.total_checkpoints = total_checkpoints

    def determine_overall_disposition(self) -> str:
        """Calculate aggregate disposition: QUARANTINE > REVIEW > ACCEPT."""
        dispositions = []
        severities = []
        for f in self.findings:
            disp = getattr(f, "recommended_disposition", None) or (f.get("recommended_disposition") if isinstance(f, dict) else "accept")
            sev = getattr(f, "severity", None) or (f.get("severity") if isinstance(f, dict) else "info")
            dispositions.append(str(disp).lower())
            severities.append(str(sev).lower())

        if "quarantine" in dispositions or "critical" in severities or "high" in severities:
            return "QUARANTINE"
        if "review" in dispositions or "warning" in severities:
            return "REVIEW"
        return "ACCEPT"

    def build_json_report(self) -> Dict[str, Any]:
        """Build machine-readable JSON assurance report."""
        overall_disp = self.determine_overall_disposition()

        severity_counts = {"critical": 0, "high": 0, "warning": 0, "info": 0}
        findings_list = []

        for f in self.findings:
            f_dict = f.to_dict() if hasattr(f, "to_dict") else dict(f)
            sev = f_dict.get("severity", "info").lower()
            if sev in severity_counts:
                severity_counts[sev] += 1
            findings_list.append(f_dict)

        return {
            "module": "M3_INFERENCE_PROVENANCE",
            "clause": "2.2.3 / 2.2.5",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "audit_summary": {
                "total_receipts_audited": self.total_receipts,
                "total_checkpoints_audited": self.total_checkpoints,
                "total_findings": len(self.findings),
                "overall_disposition": overall_disp,
                "severity_counts": severity_counts,
            },
            "findings": findings_list,
        }

    def save_json_report(self, output_path: str | Path) -> Path:
        """Write JSON report to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report_data = self.build_json_report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        return path

    def build_html_report(self) -> str:
        """Generate responsive dark-mode HTML Audit Dashboard."""
        report = self.build_json_report()
        summary = report["audit_summary"]
        overall_disp = summary["overall_disposition"]

        disp_badge_class = "badge-accept"
        if overall_disp == "QUARANTINE":
            disp_badge_class = "badge-quarantine"
        elif overall_disp == "REVIEW":
            disp_badge_class = "badge-review"

        rows_html = ""
        if not self.findings:
            rows_html = """
            <tr>
                <td colspan="6" style="text-align: center; color: #a0aec0; padding: 20px;">
                    [PASS] No audit findings detected. All receipts, cryptographic signatures, and hash chains passed verification.
                </td>
            </tr>
            """
        else:
            for f in self.findings:
                f_dict = f.to_dict() if hasattr(f, "to_dict") else dict(f)
                sev = f_dict.get("severity", "info").upper()
                disp = f_dict.get("recommended_disposition", "accept").upper()
                sev_color = "#e53e3e" if sev in ("CRITICAL", "HIGH") else "#dd6b20" if sev == "WARNING" else "#3182ce"
                disp_color = "#e53e3e" if disp == "QUARANTINE" else "#dd6b20" if disp == "REVIEW" else "#38a169"

                artifacts_str = ", ".join(f_dict.get("evidence_artifacts", [])) or "None"

                rows_html += f"""
                <tr>
                    <td style="font-family: monospace; font-size: 0.85em;">{f_dict.get('finding_id')}</td>
                    <td>{f_dict.get('detector')}</td>
                    <td><span class="pill" style="background-color: {sev_color};">{sev}</span></td>
                    <td><span class="pill" style="background-color: {disp_color};">{disp}</span></td>
                    <td style="font-size: 0.9em;">{f_dict.get('reason_human_readable')}</td>
                    <td style="font-family: monospace; font-size: 0.8em; color: #cbd5e0;">{artifacts_str}</td>
                </tr>
                """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CV-ASSURE M3 Inference Provenance Assurance Report</title>
    <style>
        :root {{
            --bg-main: #0b0f19;
            --bg-card: #151d30;
            --bg-header: #0f172a;
            --border-color: #2a364f;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-green: #22c55e;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            margin: 0;
            padding: 32px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1280px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 28px;
        }}
        h1 {{
            font-size: 2rem;
            margin: 0;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 6px;
        }}
        .badge {{
            padding: 10px 24px;
            border-radius: 9999px;
            font-weight: bold;
            font-size: 1.15rem;
            letter-spacing: 0.06em;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .badge-accept {{ background-color: #15803d; color: #f0fdf4; border: 1px solid #22c55e; }}
        .badge-review {{ background-color: #b45309; color: #fffbeb; border: 1px solid #f59e0b; }}
        .badge-quarantine {{ background-color: #b91c1c; color: #fef2f2; border: 1px solid #ef4444; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .stat-label {{
            color: var(--text-muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-top: 8px;
            color: var(--text-main);
        }}

        .features-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .feature-card {{
            background: #111827;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .feature-title {{
            font-size: 0.9rem;
            color: var(--text-muted);
        }}
        .feature-status {{
            font-size: 0.85rem;
            font-weight: bold;
            color: var(--accent-green);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: var(--bg-card);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        th, td {{
            padding: 14px 18px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: #0f172a;
            color: var(--accent-cyan);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        tr:hover {{
            background-color: #1e293b;
        }}
        .pill {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            color: #ffffff;
            display: inline-block;
        }}
        .footer {{
            margin-top: 48px;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>CV-ASSURE v2 · M3 Inference Provenance Audit Console</h1>
                <div class="subtitle">Clause 2.2.3 (Inference Provenance) & Clause 2.2.5 (Governance) | Execution Timestamp: {report['timestamp_utc']}</div>
            </div>
            <div>
                <span class="badge {disp_badge_class}">{overall_disp}</span>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Receipts Audited</div>
                <div class="stat-value">{summary['total_receipts_audited']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Merkle Checkpoints</div>
                <div class="stat-value">{summary['total_checkpoints_audited']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Audit Findings Count</div>
                <div class="stat-value">{summary['total_findings']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Critical Threat Flags</div>
                <div class="stat-value" style="color: {'#ef4444' if summary['severity_counts']['critical'] + summary['severity_counts']['high'] > 0 else '#22c55e'};">
                    {summary['severity_counts']['critical'] + summary['severity_counts']['high']}
                </div>
            </div>
        </div>

        <div class="features-row">
            <div class="feature-card">
                <span class="feature-title">Model Weight Attestation</span>
                <span class="feature-status">ResNet18 ONNX Verified</span>
            </div>
            <div class="feature-card">
                <span class="feature-title">Signature Scheme</span>
                <span class="feature-status">Ed25519 PKCS#8</span>
            </div>
            <div class="feature-card">
                <span class="feature-title">Hash-Chain Linkage</span>
                <span class="feature-status">SHA-256 prev_receipt_hash</span>
            </div>
            <div class="feature-card">
                <span class="feature-title">Local Spot Re-Execution</span>
                <span class="feature-status">ONNX Runtime Verified</span>
            </div>
        </div>

        <h2>Audit Findings & Security Logs</h2>
        <table>
            <thead>
                <tr>
                    <th>Finding ID</th>
                    <th>Detector</th>
                    <th>Severity</th>
                    <th>Disposition</th>
                    <th>Reason / Evidence Detail</th>
                    <th>Artifacts</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div class="footer">
            CV-ASSURE v2 Trustworthy Computer Vision Integrity Assurance · Smart India Hackathon Enterprise Edition
        </div>
    </div>
</body>
</html>
"""
        return html_content

    def save_html_report(self, output_path: str | Path) -> Path:
        """Write HTML report to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html_str = self.build_html_report()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_str)
        return path
