from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtCore import QUrl
import sys
from dashboard import Dashboard
from utils import *


if __name__ == '__main__':

    Path(TARGET_DIR).mkdir()
    Path(CACHE_DIR).mkdir()

    clear_target()
    clear_cache()

    # overwrite automated QApplication from ApplicationContext to include flags
    ApplicationContext.app = QApplication(sys.argv)

    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext

    main_window = QMainWindow()

    db = Dashboard()
    db.show()

    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)