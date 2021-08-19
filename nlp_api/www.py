from webs.controllers.index import router_index
from application import app 

app.register_blueprint(router_index, url_prefix = "/")

@app.errorhandler(404)
def page_not_found(err):
    print(err)
    return "this page not exist", 404

