
#!/usr/bin/env python3
# Placeholder smoke: checks for README.md and prints size.
import pathlib, sys, traceback
LOG = pathlib.Path('logs')/ 'smoke_http.log'
LOG.parent.mkdir(parents=True, exist_ok=True)
try:
    readme = pathlib.Path('README.md')
    msg = f"README.md exists={readme.exists()} size={(readme.stat().st_size if readme.exists() else 0)}"
    LOG.write_text(msg, encoding='utf-8')
    print(msg)
except Exception as e:
    LOG.write_text('SMOKE ERROR: ' + ''.join(traceback.format_exception(e)), encoding='utf-8')
    print(f"SMOKE ERROR: {e}")
    sys.exit(0)
