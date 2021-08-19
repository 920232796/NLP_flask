from application import app 

from application import manager
from flask_script import Server
import www

manager.add_command("runserver", Server(host="0.0.0.0", port=app.config["SERVER_PORT"], use_debugger=True, use_reloader=True))

def main():
    manager.run()

if __name__ == "__main__":
    main()