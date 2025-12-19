from pathlib import Path
import subprocess
import sys
import re

# Locate repo root by searching for CITATION.cff
HERE = Path(__file__).resolve()
for parent in HERE.parents:
    cff = parent / "CITATION.cff"
    if cff.exists():
        ROOT = parent
        CFF_FILE = cff
        break
else:
    raise RuntimeError("Could not find CITATION.cff in parent directories")

OUT_FILE = ROOT / "docs" / "source" / "citation.md"

cmd = [
    "cffconvert",
    "--infile", str(CFF_FILE),
    "--format", "bibtex",
]

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print("cffconvert failed:", file=sys.stderr)
    print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)

bibtex = result.stdout.strip()

year_match = re.search(r"^\s*year\s*=\s*\{(\d{4})\}", bibtex, flags=re.MULTILINE)
year = year_match.group(1) if year_match else "0000"

bibtex = re.sub(r"^(@\w+\{)[^,]+,", rf"\1socca{year},", bibtex, count=1, flags=re.MULTILINE)

lines = bibtex.splitlines()
field_info = [] 
pattern = re.compile(r"^(\s*)([A-Za-z][\w-]*)\s*=\s*(.+?)(,?)\s*$")
for i, line in enumerate(lines):
    m = pattern.match(line)
    if m:
        indent, key, value, comma = m.groups()
        field_info.append((i, indent, key, value.strip(), comma))

if field_info:
    max_key = max(len(key) for (_, _, key, _, _) in field_info)
    for i, indent, key, value, comma in field_info:
        comma_txt = "," if comma else ""
        lines[i] = f"  {indent}{key.rjust(max_key)} = {value}{comma_txt}"
    bibtex = "\n".join(lines)

OUT_FILE.write_text(
    "# Citation\n\n"
    "If you are going to include in a publication any results obtained using **``socca``**, "
    "please consider adding an hyperlink to the [GitHub repository](https://github.com/lucadimascolo/socca) "
    "or citing it as follows:\n\n"
    "```bibtex\n"
    f"{bibtex}\n"
    "```\n"
    "```{note}\n"
    "In the coming months, we plan to submit a dedicated paper to the Journal of Open Source Software (JOSS). "
    "Once available, we will update this section with the relevant citation information.\n"
    "```\n",
    encoding="utf-8",
)

print("✓ citation.md generated")
