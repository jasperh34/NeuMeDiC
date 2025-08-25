import sys
from PyQt5.QtWidgets import QApplication

from new_dashboard import FitbitDashboard

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dash = FitbitDashboard()
    dash.show()
    sys.exit(app.exec_())
