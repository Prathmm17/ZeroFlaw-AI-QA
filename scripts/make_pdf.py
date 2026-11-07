import subprocess
from pathlib import Path

chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
html_path = Path(r"C:\Users\hp\Desktop\Vision Guard\docs\MENTOR_JUDGE_ANALYSIS.html").resolve()
pdf_path = Path(r"C:\Users\hp\Desktop\Vision Guard\docs\MENTOR_JUDGE_ANALYSIS.pdf").resolve()

cmd = [
    chrome,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    html_path.as_uri()
]

print("Running command:", cmd)
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
print("Stdout:", res.stdout)
print("Stderr:", res.stderr)
print("PDF exists:", pdf_path.exists())
if pdf_path.exists():
    print("PDF size:", pdf_path.stat().st_size, "bytes")

