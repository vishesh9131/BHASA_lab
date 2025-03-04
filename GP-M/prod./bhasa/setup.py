from setuptools import setup, find_packages

# List of requirements
requirements = [
    'causal_conv1d==1.5.0.post8',
    'einops==0.8.1',
    'flash_attn==2.7.4.post1',
    'huggingface_hub==0.22.2',
    'packaging==24.2',
    'transformers==4.39.3'
]

setup(
    name='bhasa',
    version='0.1.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Vishesh Yadav and Team',
    author_email='sciencely98@gmail.com',
    description='A short description of your library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vishesh9131/BHASA_lab.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
