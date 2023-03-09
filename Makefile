.PHONY: serve

serve:
	# I know nothing about ruby, pls help
	bundle exec jekyll serve -w --livereload --port 12955 "--host=0.0.0.0"
