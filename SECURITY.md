# Security policy

## Scope

truescore reads local files, computes statistics and writes reports. It makes no network
calls and holds no credentials, and it executes nothing from its input. The dependencies
are numpy and scipy.

The realistic risk surface is parsing untrusted CSV or JSON Lines. Generated reports are
plain text and are not escaped for HTML, so treat a report as untrusted input if you
render it in a browser.

## Reporting

Open a [security advisory](https://github.com/SaifPunjwani/truescore/security/advisories/new)
instead of a public issue.

## Statistical bugs

An interval that doesn't cover at its nominal rate isn't a security issue. It is still the
most serious kind of bug this project can have. File those as regular issues with a
reproduction and they'll be prioritised.

## Supported versions

The latest PyPI release gets fixes. Older versions don't.
