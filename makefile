# INSTALLDIR
all:
	cd src/lbfgsb; make install
	cd src/fortlib; make install

install: all
	python setup.py install --record setup_files.log

clean:
	cd src/lbfgsb; make clean
	cd src/fortlib; make clean
	rm -f smili/*.pyf
	rm -f smili/*.pyc
	rm -rf smili/*.so*
	rm -f smili/*/*.pyf
	rm -f smili/*/*.pyc
	rm -rf smili/*/*.so*
	rm -f smili/*/*/*.pyf
	rm -f smili/*/*/*.pyc
	rm -rf smili/*/*/*.so*

uninstall: clean
	cd src/lbfgsb; make uninstall
	cd src/fortlib; make uninstall
	rm -rf autom4te.cache
	rm -f config.log
	rm -f config.status
	rm -f aclocal.m4
	rm makefile
	cat setup_files.log|grep smili|xargs rm -rf
	rm -rf setup_files.log *egg* *build* *dist*
