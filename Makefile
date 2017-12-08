clean-build:
	rm -rf ./dist_workflow
	mkdir dist_workflow
	cp ./main.py ./dist_workflow
	zip -r ./dist_workflow/jobs.zip ./cleaning
	zip -r ./dist_workflow/classification.zip ./classification
	zip -r ./dist_workflow/cleaning.zip ./cleaning
	zip -r ./dist_workflow/examples.zip ./examples
	zip -r ./dist_workflow/semisupervised.zip ./semisupervised
	zip -r ./dist_workflow/shared.zip ./shared
