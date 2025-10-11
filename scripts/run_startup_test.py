import traceback

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# ensure repo root is in sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print('Running controlled startup test...')
try:
    from backend.src.main import initialize_connections_and_settings
    driver = initialize_connections_and_settings()
    print('initialize_connections_and_settings() returned:', type(driver))
    # If driver has verify_connectivity method, call a lightweight check
    try:
        if hasattr(driver, 'verify_connectivity'):
            print('Calling driver.verify_connectivity()')
            driver.verify_connectivity()
            print('driver.verify_connectivity() OK')
    except Exception as e:
        print('driver.verify_connectivity() failed:', e)

    # Close driver if possible
    try:
        driver.close()
        print('Driver closed')
    except Exception:
        pass

except Exception as e:
    print('Startup test failed with exception:')
    traceback.print_exc()
    raise
