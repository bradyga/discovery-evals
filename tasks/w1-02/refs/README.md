# References

DOIs, licences and fetch notes. **No PDFs are bundled here**, and no paper text,
figure or table is reproduced anywhere in this task directory. The functional
forms are restated with attribution in `../spec.md`, section `## Method`.

## The model

D. Gobbo, P. Ballone and B. D. Garabato,
*Coarse-Grained Model of Entropy-Driven Demixing*,
**J. Phys. Chem. B** 2020, **124**(41), 9267–9274.
DOI: `10.1021/acs.jpcb.0c07575` — https://doi.org/10.1021/acs.jpcb.0c07575
PMID: 33016071.

Published by the American Chemical Society. **Not open access and not
redistributable.** Cited by DOI only.

**Fetch note.** Retrieved 2026-09-04. The publisher's article page returned
HTTP 403 and PubMed Central holds no full text; only the abstract and the
bibliographic record were read, from PubMed record 33016071. **The body of this
paper was not obtained, and nothing in `../spec.md` is taken from it.** Recorded
rather than glossed, because it bounds what this task may claim: the task
compares two functional forms as a methodological question, and does not assert
anything about which form any particular implementation of this model uses.

## Form `"arctan"`

N. C. Forero-Martinez, R. Cortes-Huerto, A. Benedetto and P. Ballone,
*Thermoresponsive Ionic Liquid/Water Mixtures: From Nanostructuring to Phase
Separation*, **Molecules** 2022, **27**, 1647.
DOI: `10.3390/molecules27051647` — https://doi.org/10.3390/molecules27051647
Open access at PubMed Central: `PMC8912101`. PMID: 35268747.

Licence: **CC BY 4.0** (MDPI). Redistributable with attribution. It is
nevertheless **not** bundled here: the task needs the equation, which `spec.md`
restates, not the article.

**Fetch note.** Retrieved 2026-09-04 from PubMed Central, article `PMC8912101`,
HTML full text. What was read: the passage of Section 2 surrounding Eqs. 15–17,
and the licence statement. The frequency law is Eq. 17 and was read verbatim at
build time.

## Form `"rational"`

M. Iannuzzi, A. Laio and M. Parrinello,
*Efficient Exploration of Reactive Potential Energy Surfaces Using
Car-Parrinello Molecular Dynamics*,
**Phys. Rev. Lett.** 2003, **90**, 238302.
DOI: `10.1103/PhysRevLett.90.238302` — https://doi.org/10.1103/PhysRevLett.90.238302
PMID: 12857293.

Published by the American Physical Society. **Not open access.** Cited by DOI
only.

**Fetch note, and a provenance limit stated rather than glossed.** Retrieved
2026-09-04: the bibliographic record was confirmed from PubMed record 12857293
and the publisher listing. **The full text was not obtained**, so the rational
switching function's form was not read in its primary source at build time; the
form as stated in `../spec.md` is the one attributed to this reference throughout
the enhanced-sampling literature.

Consequence, and why the task does not depend on the attribution being exact:
`../spec.md` **states the form explicitly**, and pins the exponent pair `(p, 2p)`,
the use of `n` and `n0` in place of an interatomic distance and its reference,
and the calibration `p = 4 n0 alpha / pi` as **this task's own**. The
specification under test is the one in `spec.md`. The citation records where the
functional form comes from; it is not load-bearing for anything the session is
asked to establish.

## What is not here

- No paper text, figure, table or PDF, from any of the three.
- No reference implementation of either form, and no reference value for any
  quantity in this task.
