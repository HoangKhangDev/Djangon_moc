from django.contrib import admin

# Register your models here.
from .models import ThuatToan_Data, User,Upload_File

admin.site.register(Upload_File)
admin.site.register(Data_ThuatToan)
admin.site.register(User)
