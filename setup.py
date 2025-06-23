from setuptools import setup, find_packages

setup(
    name="transport_frames",  # Имя пакета, оно же будет использоваться при установке
    version="0.1.0",  # Версия пакета
    description="A package for transportation data processing and frames management",
    author="Polina Krupenina",  # Ваше имя как автора
    author_email="krupenina.p@mail.com",  # Ваш email
    packages=find_packages(),  # Автоматически найдёт все подмодули в директории
    install_requires=[  # Зависимости
        "geopandas>=0.14,<2.0",
        "pandas>=2.2.0,<3.0",
        "shapely==2.1.1",
        "iduedu==0.5.7",
        "loguru>=0.7.3,<0.8.0",
        "momepy==0.6.0",
        "networkit==11.0",
        "networkx>=3.4.2,<4.0.0",
        "numpy==2.1.3",
        # "numpy>=2.1.3,<3.0",
        "osmnx>=2.0.1,<3.0.0",
        "pandera==0.20.3",
        "pydantic==2.7.3",
        "requests==2.32.3",
        "tqdm>=4.67.0,<5.0.0"
    ],
    python_requires='>=3.10,<=3.13',  # Поддержка Python 3.10 и 3.11
)