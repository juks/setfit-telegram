class Responder:
    def respond(self, cat_name, msg, links=[]):
        cat_handler = 'h_' + cat_name
        func = getattr(self, cat_handler) if hasattr(self, cat_handler) else False

        if func:
            return func(msg, links)
        else:
            return False