from setuptools import setup, find_packages

setup(
    name="transport_frames",  # Имя пакета, оно же будет использоваться при установке
    version="0.1.0",  # Версия пакета
    description="A package for transportation data processing and frames management",
    author="Polina Krupenina",  # Ваше имя как автора
    author_email="krupenina.p@mail.com",  # Ваш email
    packages=find_packages(),  # Автоматически найдёт все подмодули в директории
    install_requires=[  # Зависимости
        "geopandas==0.14.4",
        "pandas==2.2.2",
        "shapely==2.0.5",
        "iduedu==0.1.5",
        "loguru==0.7.2",
        "momepy==0.6.0",
        "networkit==11.0",
        "networkx==3.3",
        "numpy==1.26.4",
        "osmnx==1.9.4",
        "pandera==0.20.3",
        "pydantic==2.7.3",
        "requests==2.32.3",
        "tqdm==4.66.5"
    ],
    python_requires='>=3.10,<=3.13',  # Поддержка Python 3.10 и 3.11
)