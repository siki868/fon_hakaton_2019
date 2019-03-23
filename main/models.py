from django.db import models

# Create your models here.
class TempModel(models.Model):
    temp_name = models.CharField(max_length = 100)

    def __str__(self):
        return self.temp_name

    class Meta:
        verbose_name_plural = "Temp Models"
