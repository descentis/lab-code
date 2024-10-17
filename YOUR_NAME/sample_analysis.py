import kdap

knol = kdap.knol()

#knol.download_dataset(sitename='wikipedia', article_list=['Vector', 'Derivative'], destdir='/Users/amitverma/Documents/research')

knol.get_num_instances(dir_path="/Users/amitverma/Documents/research", granularity="yearly", start="2015-01-01")