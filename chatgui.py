from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from chatapp import ChatApp as cA

class ChatBotUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ChatBot - SL")
        self.setGeometry(100, 100, 500, 600)

        # Layout for the entire window
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        # ChatLog for displaying messages
        self.chatLog = QtWidgets.QTextBrowser(self)
        self.chatLog.setOpenExternalLinks(True)
        self.chatLog.setStyleSheet("""
            background-color: #FFFFFF;
            color: #000000;
            font: 12pt Arial;
            padding: 10px;
            border: none;
        """)
        self.layout.addWidget(self.chatLog)

        # EntryBox for typing messages
        self.entryBox = QtWidgets.QTextEdit(self)
        self.entryBox.setFixedHeight(90)
        self.entryBox.setStyleSheet("""
            background-color: #F5F5F5;
            color: #000000;
            font: 12pt Arial;
            padding: 10px;
            border: 1px solid #E0E0E0;
            border-radius: 5px;
        """)
        self.layout.addWidget(self.entryBox)

        # SendButton to send messages
        self.sendButton = QtWidgets.QPushButton("Send", self)
        self.sendButton.setStyleSheet("""
            background-color: #32de97;
            color: #ffffff;
            font: bold 12pt Verdana;
            height: 40px;
            border-radius: 20px;
            margin-top: 10px;
        """)
        self.sendButton.clicked.connect(self.send)
        self.layout.addWidget(self.sendButton)

        # Show introduction message
        self.show_intro()

    def show_intro(self):
        intro_message = '''
            <div style="text-align: left; padding: 10px; border-radius: 10px; background-color: #E0F7FA; color: #00796B;">
                <b>Bot:</b> Hi! This is the movie recommendation chatbot, here to help you with movie recommendations.
            </div>
        '''
        self.chatLog.append(intro_message)
        self.chatLog.append('<br>')  # Add a single empty line after the intro

    def send(self):
        msg = self.entryBox.toPlainText().strip()
        self.entryBox.clear()
        if msg != '':
            # Format user message for right alignment
            user_msg_html = f'''
                <div style="text-align: right; padding: 10px; border-radius: 10px; background-color: #DCF8C6; color: #000000; margin: 5px;">
                    <b>You:</b> {msg}
                </div>
            '''
            # Append user message with a single newline
            self.chatLog.append(user_msg_html)
            
            # Get bot response and format for left alignment
            res = cA().chatbot_response(msg)
            bot_msg_html = f'''
                <div style="text-align: left; padding: 10px; border-radius: 10px; background-color: #E0F7FA; color: #00796B; margin: 5px;">
                    <b>Bot:</b> {res}
                </div>
            '''
            # Append bot response with a single newline
            self.chatLog.append(bot_msg_html)
            
            # Ensure only a single <br> is added between messages
            self.chatLog.append('<br>')
            
            # Scroll to the end of chatLog
            self.chatLog.moveCursor(QtGui.QTextCursor.End)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWin = ChatBotUI()
    mainWin.show()
    sys.exit(app.exec_())
