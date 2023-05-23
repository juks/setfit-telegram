from lib.responder import Responder

class MyResponder(Responder):
    def h_courier_app_problem(self, msg, links=[]):
        if not len(links):
            return 'Перезагрузите телефон.'
        else:
            return False

    def h_open_shift(self, msg, links=[]):
        if not len(links):
            return 'Создали задачу про открытие смены https://st.yandex-team.ru/MARKETTPLSUP-110972'

    def h_close_shift(self, msg, links=[]):
        if not len(links):
            return 'Создали задачу про закрытие смены https://st.yandex-team.ru/MARKETTPLSUP-110981'

    def h_version(self, msg, links=[]):
        if not len(links):
            return 'Текущая версия приложения 5.2.2. Вот ссылка на APK.'
        else:
            return False

    def h_attention(self, msg, links=[]):
        if len(links):
            return 'Займёмся.'
        else:
            return False