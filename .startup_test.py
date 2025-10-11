import traceback

try:
    from backend.src.main import initialize_connections_and_settings
    print('Calling initialize_connections_and_settings()...')
    res = initialize_connections_and_settings()
    print('initialize_connections_and_settings returned:', res)
except Exception as e:
    traceback.print_exc()
    print('INITIALIZE_FAIL')
