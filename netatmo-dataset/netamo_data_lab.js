#
# netatmo Data
#

# from terminal
#
# start mongo database
sudo service mongod start


# import dataset
mongoimport --db netatmo --collection iot_meteo -v --file /home/master/dataset/netatmo/netatmo_data_milan.json

# access to mongo shell
mongo


#
# Start script
#
use netatmo
db.iot_meteo.count({});

db.iot_meteo.find({});



# mark each sensor type
db.iot_meteo.update({'measures.measures.type.0': 'temperature'}, {$set: {'sensorType':'temperature'}}, {multi:true});
db.iot_meteo.update({'measures.measures.rain_60min': {$exists: true}},{$set: {'sensorType':'rain'}},{multi:true});
db.iot_meteo.update({'measures.measures.wind_strength': {$exists: true}}, {$set: {'sensorType':'wind'}},{multi:true});


# group by sensor type
db.iot_meteo.aggregate([ {$group: {_id:'$sensorType',total: {$sum: 1}}}]);


# only temperature sensor
# group by sensor type
db.iot_meteo.aggregate([ {$match: {"sensorType": "temperature" }}, {$group: {_id:"$sensorType",total: {$sum: 1}}}]);

# avg temperature
db.iot_meteo.aggregate([
    {$match: {sensorType: 'temperature'}},
    {$unwind: '$measures.measures.res.value'},
    {$group: {_id: '$sensorType', total: {$avg: '$measures.measures.res.value'}}}
    ])

# where rain_live is positive
db.iot_meteo.aggregate([
    {$match: {sensorType: 'rain', 'measures.measures.rain_24h': {$gt: 0}}},
    {$group: {_id: '$sensorType', total: {$avg: '$measures.measures.rain_24h'}}}
    ])


# extract data and create my first map
db.iot_meteo.find({sensorType: 'rain', 'measures.measures.rain_24h': {$gt: 0}}).forEach(function(x){
    var point = x;
    point.location = new Object();
    point.location.type = "Point";
    point.location.coordinates = x.place.location;
    db.rain.insertOne(point);
});

db.rain.createIndex( { location : "2dsphere" } )

# create temperature map
db.iot_meteo.find({sensorType: 'temperature'}).forEach(function(x){
    var point = x;
    point.location = new Object();
    point.location.type = "Point";
    point.location.coordinates = x.place.location;
    db.temperature.insertOne(point);
});

db.temperature.createIndex( { location : "2dsphere" } )


# find data near Bergamo
db.temperature.find({
    location: {
        $nearSphere: {
            $geometry: {
                type: 'Point',
                coordinates: [9.665848, 45.698018]
                },
                $maxDistance: 5000
            }
        }
    }).forEach(function (x){
        db.temperature_bergamo.insertOne(x);
    })


# find our sensor
db.temperature.aggregate([{
    $geoNear: {
        near: {
            type: 'Point',
            coordinates: [9.231507, 45.480525]
        },
        spherical: true,
        maxDistance: 1000,
        distanceField: 'distanceFromSF'
    }
}])


# find our sensor
db.temperature.aggregate([{
    $geoNear: {
        near: {
            type: 'Point',
            coordinates: [9.231507, 45.480525]
        },
        spherical: true,
        maxDistance: 1000,
        distanceField: 'distanceFromSF'
    }
}]).forEach(function(x) {
    db.temperature_cnr.insertOne(x);
})

# avg of our sensors
db.temperature.aggregate([{
    $geoNear: {
        near: {
            type: 'Point',
            coordinates: [9.231507, 45.480525]
        },
        spherical: true,
        maxDistance: 1000,
        distanceField: 'distanceFromSF'
    }},
    {$unwind: '$measures.measures.res.value'},
    {$group: {_id: '$sensorType', total: {$avg: '$measures.measures.res.value'}}}
])



#
# from shell
#
# import dataset
mongoimport --db netatmo --collection quartieri_milano -v --file /home/master/dataset/netatmo/quartieri_milano.json
mongo
use netatmo
db.quartieri_milano.createIndex( { geometry : "2dsphere" } )


db.temperature.find({}).forEach(function(x) {
    var q = db.quartieri_milano.findOne({ geometry: { $geoIntersects: { $geometry: { type: "Point", coordinates: x.location.coordinates } } } });
    if(q) {
        print(q.NIL)
        x.quartiere = q.NIL;
        x.quartiere_geometry = q.geometry;
        db.temperature_milano.insertOne(x);
    }
    
})
db.temperature_milano.createIndex( { quartiere_geometry : "2dsphere" } )

# avg of temperature for each quartiere
db.temperature_milano.aggregate([
    {$unwind: '$measures.measures.res.value'},
    {$group: {_id: '$quartiere', total: {$avg: '$measures.measures.res.value'}}}
])