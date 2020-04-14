from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QApplication
import sys
from elements import ToolboxWindow


if __name__ == '__main__':

    # overwrite automated QApplication from ApplicationContext to include flags
    ApplicationContext.app = QApplication(sys.argv)

    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext

    window = ToolboxWindow()
    window.show()

    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)