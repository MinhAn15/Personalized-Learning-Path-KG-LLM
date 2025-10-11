import traceback

try:
    from backend.src import api, config, content_generator, data_loader, main, path_generator, prepare_data, recommendations, session_manager
    from backend.src.knowledge_extractor import extractor, main as extractor_main, validator
    print('IMPORT_OK')
except Exception:
    traceback.print_exc()
    print('IMPORT_FAIL')
