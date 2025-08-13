## Note for OpenML dataset 1477 users

If you experience problems accessing **OpenML dataset 1477** online,  
you can use the attached copy of the dataset by placing it into your local  
OpenML cache directory.

### Steps

1. Locate your local OpenML cache directory by running in Python:
   ```python
   import openml
   print(openml.config.cache_directory)

2. Inside the cache directory, ensure there is a subfolder named datasets. If it does not exist, create it manually.

3. Copy the provided folder 1477 (including all its contents) into the datasets subfolder.

4. Rerun the example — OpenML will now load the dataset from the local cache instead of downloading it.
