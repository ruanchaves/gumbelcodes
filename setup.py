from setuptools import setup

setup(name='gumbelcodes',
      version='0.1',
      description='Neural compression for word embeddings',
      long_description='Based on the original neuralcompressor repository ( https://github.com/zomux/neuralcompressor ).',
      classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Text Processing :: Linguistic',
      ],
      keywords='NLP embeddings compression',
      url='https://github.com/ruanchaves/gumbelcodes',
      author='Ruan Chaves Rodrigues',
      author_email='ruanchaves93@gmail..com',
      license='MIT',
      packages=['gumbelcodes'],
      install_requires=[
        "aiocontextvars==0.2.2"
        "contextvars==2.4"
        "immutables==0.11"
        "loguru==0.4.0"
        "marisa-trie==0.7.5"
        "numpy==1.18.0"
        "scipy==1.4.1"
        "six==1.13.0"
      ],
      extras_require={
            'training': [
                "absl-py==0.9.0"
                "astor==0.8.1"
                "gast==0.3.2"
                "google-pasta==0.1.8"
                "grpcio==1.26.0"
                "h5py==2.10.0"
                "Keras-Applications==1.0.8"
                "Keras-Preprocessing==1.1.0"
                "Markdown==3.1.1"
                "protobuf==3.11.2"
                "tensorboard==1.14.0"
                "tensorflow-estimator==1.14.0"
                "tensorflow-gpu==1.14.0"
                "termcolor==1.1.0"
                "Werkzeug==0.16.0"
                "wrapt==1.11.2",                
            ]
      },
      include_package_data=True,
      zip_safe=False)
