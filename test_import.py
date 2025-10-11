import sys, os, traceback
print('CWD:', os.getcwd())
print('First entries in sys.path:', sys.path[:5])
try:
    import src
    print('Imported src package from', getattr(src, '__file__', 'package without __file__'))
    from src import Config
    print('Config loaded. STUDENT_FILE =', Config.STUDENT_FILE)
except Exception as e:
    print('Import error:')
    traceback.print_exc()
    sys.exit(1)
print('IMPORT_OK')
