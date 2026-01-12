"""Generate citation.md file from CITATION.cff using cffconvert."""

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
    "--infile",
    str(CFF_FILE),
    "--format",
    "bibtex",
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

year_match = re.search(
    r"^\s*year\s*=\s*\{(\d{4})\}", bibtex, flags=re.MULTILINE
)
year = year_match.group(1) if year_match else "0000"

bibtex = re.sub(
    r"^(@\w+\{)[^,]+,", rf"\1socca{year},", bibtex, count=1, flags=re.MULTILINE
)

lines = bibtex.splitlines()
field_info = []
pattern = re.compile(r"^(\s*)([A-Za-z][\w-]*)\s*=\s*(.+?)(,?)\s*$")
for i, line in enumerate(lines):
    m = pattern.match(line)
    if m:
        indent, key, value, comma = m.groups()
        field_info.append((i, indent, key, value.strip(), comma))

if field_info:
    desired_order = ["author", "title", "year", "month", "url"]
    order_index = {k: i for i, k in enumerate(desired_order)}

    ordered_fields = sorted(
        field_info, key=lambda t: (order_index.get(t[2].lower(), 1000), t[0])
    )

    max_key = max(len(key) for (_, _, key, _, _) in ordered_fields)

    header_line = next(
        (ln for ln in lines if ln.strip().startswith("@")), None
    )
    closing_brace = "}"

    rebuilt_field_lines = []
    for idx, (_, indent, key, value, _) in enumerate(ordered_fields):
        comma_txt = "," if idx < len(ordered_fields) - 1 else ""
        rebuilt_field_lines.append(
            f"  {indent}{key.rjust(max_key)} = {value}{comma_txt}"
        )

    if header_line:
        bibtex = "\n".join(
            [header_line] + rebuilt_field_lines + [closing_brace]
        )
    else:
        bibtex = "\n".join(lines)

OUT_FILE.write_text(
    "# Citing socca\n\n"
    "If you are going to include in a publication any results obtained "
    "using **``socca``**, please consider adding an hyperlink to the "
    "[GitHub repository](https://github.com/lucadimascolo/socca) "
    "or citing it as follows:\n\n"
    "```bibtex\n"
    f"{bibtex}\n"
    "```\n"
    "```{note}\n"
    "In the coming months, we plan to submit a dedicated paper to the "
    "Journal of Open Source Software (JOSS). Once available, we will "
    "update this section with the relevant citation information.\n"
    "```\n\n"
    "If you use the `Disk` component in your work, please also consider "
    "citing the following paper:\n"
    "```bibtex\n"
    "@article{vanAsselt2026,\n"
    "  author = {{van Asselt}, Marloes and {Rizzo}, Francesca and "
    "{Di Mascolo}, Luca},\n"
    '        title = "{Early thin-disc assembly revealed by JWST edge-on galaxies}",\n'
    "      journal = {arXiv e-prints},\n"
    "     keywords = {Astrophysics of Galaxies},\n"
    "         year = {2026},\n"
    "        month = {1},\n"
    "          eid = {arXiv:2601.03339},\n"
    "        pages = {arXiv:2601.03339},\n"
    "          doi = {10.48550/arXiv.2601.03339},\n"
    "archivePrefix = {arXiv},\n"
    "       eprint = {2601.03339},\n"
    " primaryClass = {astro-ph.GA},\n"
    "       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260103339V},\n"
    "      adsnote = {Provided by the SAO/NASA Astrophysics Data System}\n"
    "}\n"
    "```\n",
    encoding="utf-8",
)

print("✓ citation.md generated")
