build: 

	rm -rf ./dist*
	mkdir ./dist_workflow
	cp ./sample/main.py ./dist_workflow
	rsync -av --progress sample ./dist_workflow --exclude sample/main.py
	rsync -av --progress shared ./dist_workflow
	cd dist_workflow && zip -x main.py -x sample/\* -r ./shared.zip * && rm -rf shared &&  . 
	cd dist_workflow && zip -x main.py -x shared.zip i  -r ./sample.zip * && rm -rf sample && .
