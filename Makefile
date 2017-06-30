build: 

	rm -rf ./dist
	mkdir ./dist
	cp ./sample/main.py ./dist
	rsync -av --progress sample ./dist --exclude sample/main.py
	rsync -av --progress shared ./dist
	cd dist && zip -x main.py -x sample/\* -r ./shared.zip * && rm -rf shared &&  . 
	cd dist && zip -x main.py -x shared.zip i  -r ./sample.zip * && rm -rf sample && .
