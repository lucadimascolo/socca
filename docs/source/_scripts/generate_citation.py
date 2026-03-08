"""Generate citation.md file from the official NASA ADS entry."""

from pathlib import Path

# Locate repo root by searching for CITATION.cff
HERE = Path(__file__).resolve()
for parent in HERE.parents:
    if (parent / "CITATION.cff").exists():
        ROOT = parent
        break
else:
    raise RuntimeError("Could not find CITATION.cff in parent directories")

OUT_FILE = ROOT / "docs" / "source" / "citation.md"

SOCCA_BIBTEX = """\
@software{2026ascl.soft02005D,
       author = {{Di Mascolo}, Luca},
        title = "{socca: Source Characterization using a Composable Analysis}",
 howpublished = {Astrophysics Source Code Library, record ascl:2602.005},
         year = 2026,
        month = feb,
          eid = {ascl:2602.005},
archivePrefix = {ascl},
       eprint = {2602.005},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026ascl.soft02005D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}"""

DISK_BIBTEX = """\
@article{vanAsselt2026,
  author = {{van Asselt}, Marloes and {Rizzo}, Francesca and {Di Mascolo}, Luca},
        title = "{Early thin-disc assembly revealed by JWST edge-on galaxies}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies},
         year = {2026},
        month = {1},
          eid = {arXiv:2601.03339},
        pages = {arXiv:2601.03339},
          doi = {10.48550/arXiv.2601.03339},
archivePrefix = {arXiv},
       eprint = {2601.03339},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260103339V},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}"""

OUT_FILE.write_text(
    "# Citing socca\n\n"
    "If you are going to include in a publication any results obtained "
    "using **``socca``**, please consider adding an hyperlink to the "
    "[GitHub repository](https://github.com/lucadimascolo/socca) "
    "or citing it as follows:\n\n"
    "```bibtex\n"
    f"{SOCCA_BIBTEX}\n"
    "```\n\n"
    "If you use the `Disk` component in your work, please also consider "
    "citing the following paper:\n"
    "```bibtex\n"
    f"{DISK_BIBTEX}\n"
    "```\n",
    encoding="utf-8",
)

print("citation.md generated")
