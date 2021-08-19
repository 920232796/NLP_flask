from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_script import Manager
from flask_caching import Cache

class Application(Flask):
    def __init__(self, import_name):
        super().__init__(import_name)
        self.config.from_pyfile("./config/base_setting.py")
        self.config.from_pyfile("./config/local_setting.py") 


        # db.init_app(self)

# db = SQLAlchemy()
app = Application(__name__)

cache = Cache()  # 缓存
cache.init_app(app, config={'CACHE_TYPE': 'simple'})


manager = Manager(app)




 
