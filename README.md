# RealEstate_Fall2023

## purpose

## usage

### run following command and get service running

```sh
docker-compose up -d
```

### COPY .env.example to .env

```sh
cp .env.example .env
```

Edit .env and fill fields in the file

### visit http://127.0.0.1:8000/ to test if web service is running

### visit http://127.0.0.1:8000/docs#/ to see api documentation


## check house-price.ipynb to find out the training processes

## TODO

1. write tests
1. improve data preprocessing
1. evaluate model
1. plot metrics

## Conclusion

[Zillow API](https://bridgedataoutput.com/docs/explorer/reso-web-api) would be much suitable for this project becasue the data it provides is more compatible with the training dataset. In contrast, [ATTOM API](https://api.developer.attomdata.com/docs#/) provides free trail api that we can use to implement its APIs in our project. Consequently, this project is currently implementing ATTOM API to get property details.

It is recommended to use Zillow API to get better predictions.