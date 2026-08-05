"""`python -m dstparser` -- show which paths resolved, from where, and whether
they exist. See dstparser.paths.
"""

from dstparser.paths import report

if __name__ == "__main__":
    report()
