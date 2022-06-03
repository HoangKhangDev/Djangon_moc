from django.urls import path, include
import app.views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path("", app.views.index, name='home'),
    path("test/", app.views.test, name='test'),
    path("svm_imoc/", app.views.svm_imoc, name='svm_imoc'),
    path("login/", app.views.login, name='login'),
    path("logout/", app.views.logout, name='logout'),
    path("process_test/", app.views.process_test, name='process_test'),
]
