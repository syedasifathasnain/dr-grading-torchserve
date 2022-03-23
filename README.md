# dr-grading-torchserve

## Command 

```
docker-compose up
```

Then run curl command with the sample file added in the repository

```
curl --location --request POST 'http://0.0.0.0:8080/predictions/dr-score' --form 'file=@"13_left.jpeg"'
```