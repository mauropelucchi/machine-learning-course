
import com.vividsolutions.jts.geom.{Coordinate, Envelope, GeometryFactory}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.datasyslab.geospark.enums.{FileDataSplitter, GridType, IndexType}
import org.datasyslab.geospark.formatMapper.shapefileParser.ShapefileRDD
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import org.datasyslab.geospark.spatialOperator.{JoinQuery, KNNQuery, RangeQuery}
import org.datasyslab.geospark.spatialRDD.{CircleRDD, PointRDD, PolygonRDD}

import org.datasyslab.geosparksql.utils.{Adapter, GeoSparkSQLRegistrator}
import org.datasyslab.geosparkviz.core.Serde.GeoSparkVizKryoRegistrator

// register geospark sql operator
GeoSparkSQLRegistrator.registerAll(spark)

// load dataset of meteo sensor from RegioneLombardia Open Data

var stazioni = spark.read.format("csv").option("delimiter", "\t").option("header", "true").load("/home/master/dataset/meteo/Stazioni_Meteorologiche.tsv")
var dati = spark.read.format("csv").option("delimiter", "\t").option("header", "true").load("/home/master/dataset/meteo/Dati_sensori_meteo.tsv")
stazioni.printSchema
dati.printSchema
stazioni.count()
dati.count()

stazioni.createOrReplaceTempView("stazioni")
dati.createOrReplaceTempView("dati")

stazioni.show()

dati.show()

// join dataset
val dati_sensori = spark.sql("select * from stazioni inner join dati on (stazioni.IdSensore = dati.IdSensore)")
dati_sensori.count()
dati_sensori.printSchema
dati_sensori.createOrReplaceTempView("dati_sensori")


// find point
dati_sensori.select("location").show(false)
var spatialDf = spark.sql("SELECT ST_Point(CAST(lng AS Decimal(24,20)), CAST(lat AS Decimal(24,20))) AS sensor_location, dati_sensori.* FROM dati_sensori")
spatialDf.show()
spatialDf.select("location","sensor_location").show(false)
spatialDf.printSchema()


// trasform from 4326 to 3857
spatialDf.createOrReplaceTempView("spatialdf")
spatialDf = spark.sql("SELECT ST_Transform(sensor_location, 'epsg:4326', 'epsg:3857') AS new_sensor_location, spatialdf.* FROM spatialdf")
spatialDf.select("NomeStazione","location","sensor_location","new_sensor_location").show(false)

spatialDf.createOrReplaceTempView("spatialdf")
spatialDf.show()

// milan area
//{ "type": "Feature", "properties": { "STAT_LEVL_": 3, "NUTS_ID": "ITC4C", "SHAPE_Leng": 2.707079, "SHAPE_Area": 0.179000 }, "geometry": { "type": "MultiPolygon", "coordinates": [ [ [ [ 8.99650172599999, 45.576738546 ], [ 9.002092500000089, 45.574169 ], [ 9.049433500000077, 45.578163500000073 ], [ 9.064432, 45.582008502 ], [ 9.056367224000013, 45.609744677000037 ], [ 9.055403034000079, 45.613060696000048 ], [ 9.052455, 45.623199500000027 ], [ 9.052455105000092, 45.623199499 ], [ 9.052547366999988, 45.623198698000067 ], [ 9.052679895000011, 45.623197547000075 ], [ 9.107782500000013, 45.622719 ], [ 9.104852771000083, 45.614242974000035 ], [ 9.102058446000086, 45.606158688000107 ], [ 9.100149240000093, 45.600635147000048 ], [ 9.097690334000077, 45.593521263000071 ], [ 9.095737500000013, 45.5878715 ], [ 9.146128272000084, 45.586630639000077 ], [ 9.15988819399999, 45.586291804000041 ], [ 9.176286500000089, 45.585888 ], [ 9.190972015000085, 45.580246684000031 ], [ 9.20192799600008, 45.576038037000046 ], [ 9.202618769000083, 45.575772682000036 ], [ 9.226669120000082, 45.566533944000071 ], [ 9.236305907000087, 45.562832054000069 ], [ 9.253956949000013, 45.556051556 ], [ 9.258249893000084, 45.554402458000027 ], [ 9.267039937000078, 45.551025837000026 ], [ 9.305094500000081, 45.536407500000109 ], [ 9.324844337000087, 45.543164031 ], [ 9.334229079000011, 45.546374604000107 ], [ 9.359614207000078, 45.555058999000067 ], [ 9.362563469000151, 45.556067958000028 ], [ 9.36708878799999, 45.557616095000071 ], [ 9.398745, 45.568445864000068 ], [ 9.444575278000087, 45.584124660000043 ], [ 9.467666521000012, 45.592024305000109 ], [ 9.478825250999989, 45.595841769 ], [ 9.481654, 45.5968095 ], [ 9.491757626, 45.627302908000047 ], [ 9.492988556000086, 45.631017936000035 ], [ 9.495852, 45.63966 ], [ 9.495852004000085, 45.639659995000045 ], [ 9.503207100999987, 45.629792660000078 ], [ 9.531432896000013, 45.591925946 ], [ 9.538598, 45.582313500000026 ], [ 9.533597257999986, 45.56560910200011 ], [ 9.532536091999987, 45.562064400000111 ], [ 9.526755500000093, 45.542755 ], [ 9.550286999000093, 45.524517 ], [ 9.55028699799999, 45.52451699400001 ], [ 9.54089550000009, 45.501213 ], [ 9.51919150000009, 45.504474500000072 ], [ 9.483908500000013, 45.44868450000007 ], [ 9.458803663000083, 45.458856756000046 ], [ 9.434227, 45.468815 ], [ 9.425219754000011, 45.455801898 ], [ 9.415724231000013, 45.442083363000108 ], [ 9.410875, 45.43507750100008 ], [ 9.412653721000083, 45.433571604 ], [ 9.4259875, 45.422283 ], [ 9.40983463500001, 45.398416538000049 ], [ 9.404863503000087, 45.39107150500007 ], [ 9.40486350000009, 45.39107150000001 ], [ 9.401411281, 45.392967487000078 ], [ 9.391223860000082, 45.39856250400004 ], [ 9.388916, 45.39983 ], [ 9.382759827000086, 45.39465493 ], [ 9.373100500000078, 45.386535 ], [ 9.364549138000086, 45.36817860900004 ], [ 9.357994333000079, 45.354108035000081 ], [ 9.352684880999988, 45.342710743000026 ], [ 9.350357086000088, 45.337713887 ], [ 9.347544639000091, 45.331676676000029 ], [ 9.338828, 45.312965500000075 ], [ 9.32759511500015, 45.314757641000028 ], [ 9.29483145399999, 45.319984891000047 ], [ 9.250889332000071, 45.32699559800011 ], [ 9.24123076699999, 45.328536565000036 ], [ 9.237368901999986, 45.329152703000034 ], [ 9.230009720000084, 45.330326817000071 ], [ 9.201728, 45.334839 ], [ 9.187354223000085, 45.317234767 ], [ 9.18205250200009, 45.310741499000073 ], [ 9.198068613000089, 45.293991135000027 ], [ 9.200757, 45.291179500000027 ], [ 9.193098916999986, 45.293321220000109 ], [ 9.178710066000093, 45.297345319000044 ], [ 9.169327906000092, 45.299969208000107 ], [ 9.129581500000086, 45.311085 ], [ 9.090742571000078, 45.307791095000027 ], [ 9.074731493000087, 45.306433206 ], [ 9.069916830000011, 45.306024878000073 ], [ 9.036425500000092, 45.303184500000043 ], [ 9.025973735000093, 45.317570846000081 ], [ 9.022849422000093, 45.321871312000042 ], [ 9.019847000999988, 45.326003999000108 ], [ 8.991673507000087, 45.322750001000031 ], [ 8.9916735, 45.32275 ], [ 9.010436344999988, 45.29346777 ], [ 9.018697500000087, 45.280575 ], [ 9.011897425000086, 45.277611426000078 ], [ 8.996539793000011, 45.270918340000037 ], [ 8.98294150000001, 45.264992 ], [ 8.953186572000078, 45.285542928000041 ], [ 8.944611500000093, 45.291465500000072 ], [ 8.944311199999987, 45.300531765000073 ], [ 8.943877500000014, 45.3136255 ], [ 8.928849269000011, 45.319322826000075 ], [ 8.92112471100009, 45.322251270000038 ], [ 8.920407372999989, 45.322523219 ], [ 8.907376058000011, 45.327463498000043 ], [ 8.8660765, 45.343120500000026 ], [ 8.860931349000083, 45.354376905000038 ], [ 8.856379482000079, 45.364335341000071 ], [ 8.847292119000088, 45.384216399000081 ], [ 8.84289650000008, 45.393833 ], [ 8.836652074999989, 45.404249655 ], [ 8.827146260000092, 45.420106807000082 ], [ 8.818613808, 45.434340242000076 ], [ 8.81831224600009, 45.434843293000029 ], [ 8.818220235000013, 45.434996782000042 ], [ 8.815475885000012, 45.439574777000075 ], [ 8.805480261000071, 45.456249005000075 ], [ 8.789063371000083, 45.483634887000107 ], [ 8.78882198000008, 45.484037563000072 ], [ 8.787609006000082, 45.486060990000027 ], [ 8.787609, 45.486061 ], [ 8.76088221100008, 45.494968554000081 ], [ 8.758649351000088, 45.495712726000107 ], [ 8.752114545000012, 45.497890658000074 ], [ 8.722836500000085, 45.507648500000073 ], [ 8.716850586000078, 45.526655221000027 ], [ 8.712982800000077, 45.538936376000038 ], [ 8.712275392, 45.541182566000032 ], [ 8.706878499999988, 45.558319 ], [ 8.725432981000012, 45.565258714000038 ], [ 8.767663233000093, 45.581053597000107 ], [ 8.782250500000089, 45.586509500000034 ], [ 8.79400169200008, 45.582638581000026 ], [ 8.809398662000092, 45.577566719 ], [ 8.827325605999988, 45.571661467000069 ], [ 8.837002, 45.568474 ], [ 8.865976502000081, 45.575943001000041 ], [ 8.87308083100001, 45.582350372 ], [ 8.88312408200008, 45.591408348000073 ], [ 8.898034439000071, 45.604855951000047 ], [ 8.916010426000071, 45.62106843600003 ], [ 8.929746578000078, 45.633457028 ], [ 8.936001957000087, 45.639098734000072 ], [ 8.936902253999989, 45.639910710000038 ], [ 8.93997, 45.642677500000048 ], [ 8.955139290000091, 45.631706787000041 ], [ 8.960312829000088, 45.627965188000076 ], [ 8.97406550000008, 45.618019 ], [ 8.973625129999988, 45.608399531000032 ], [ 8.972677, 45.58768850000007 ], [ 8.97476321100001, 45.586729668000032 ], [ 8.99650172599999, 45.576738546 ] ] ], [ [ [ 9.487857545000082, 45.162317775000076 ], [ 9.478234500000013, 45.1616515 ], [ 9.446012515000092, 45.182744232 ], [ 9.4385435, 45.1876335 ], [ 9.465761469000086, 45.19923344800003 ], [ 9.469374296000012, 45.200773189000074 ], [ 9.470658500000013, 45.2013205 ], [ 9.508576350999988, 45.187379284000031 ], [ 9.518225879000084, 45.183831452000049 ], [ 9.526747, 45.1806985 ], [ 9.530893553999988, 45.167485204 ], [ 9.531565500000085, 45.165344 ], [ 9.51216266099999, 45.16400059800003 ], [ 9.504914765000081, 45.163498772000082 ], [ 9.487857545000082, 45.162317775000076 ] ] ] ] } }

// find all data in Milan
var milan_sensor = spark.sql(
  """
    |SELECT *
    |FROM spatialdf
    |WHERE ST_Contains (ST_GeomFromGeoJSON('{ "type": "Feature", "properties": { "STAT_LEVL_": 3, "NUTS_ID": "ITC4C", "SHAPE_Leng": 2.707079, "SHAPE_Area": 0.179000 }, "geometry": { "type": "MultiPolygon", "coordinates": [ [ [ [ 8.99650172599999, 45.576738546 ], [ 9.002092500000089, 45.574169 ], [ 9.049433500000077, 45.578163500000073 ], [ 9.064432, 45.582008502 ], [ 9.056367224000013, 45.609744677000037 ], [ 9.055403034000079, 45.613060696000048 ], [ 9.052455, 45.623199500000027 ], [ 9.052455105000092, 45.623199499 ], [ 9.052547366999988, 45.623198698000067 ], [ 9.052679895000011, 45.623197547000075 ], [ 9.107782500000013, 45.622719 ], [ 9.104852771000083, 45.614242974000035 ], [ 9.102058446000086, 45.606158688000107 ], [ 9.100149240000093, 45.600635147000048 ], [ 9.097690334000077, 45.593521263000071 ], [ 9.095737500000013, 45.5878715 ], [ 9.146128272000084, 45.586630639000077 ], [ 9.15988819399999, 45.586291804000041 ], [ 9.176286500000089, 45.585888 ], [ 9.190972015000085, 45.580246684000031 ], [ 9.20192799600008, 45.576038037000046 ], [ 9.202618769000083, 45.575772682000036 ], [ 9.226669120000082, 45.566533944000071 ], [ 9.236305907000087, 45.562832054000069 ], [ 9.253956949000013, 45.556051556 ], [ 9.258249893000084, 45.554402458000027 ], [ 9.267039937000078, 45.551025837000026 ], [ 9.305094500000081, 45.536407500000109 ], [ 9.324844337000087, 45.543164031 ], [ 9.334229079000011, 45.546374604000107 ], [ 9.359614207000078, 45.555058999000067 ], [ 9.362563469000151, 45.556067958000028 ], [ 9.36708878799999, 45.557616095000071 ], [ 9.398745, 45.568445864000068 ], [ 9.444575278000087, 45.584124660000043 ], [ 9.467666521000012, 45.592024305000109 ], [ 9.478825250999989, 45.595841769 ], [ 9.481654, 45.5968095 ], [ 9.491757626, 45.627302908000047 ], [ 9.492988556000086, 45.631017936000035 ], [ 9.495852, 45.63966 ], [ 9.495852004000085, 45.639659995000045 ], [ 9.503207100999987, 45.629792660000078 ], [ 9.531432896000013, 45.591925946 ], [ 9.538598, 45.582313500000026 ], [ 9.533597257999986, 45.56560910200011 ], [ 9.532536091999987, 45.562064400000111 ], [ 9.526755500000093, 45.542755 ], [ 9.550286999000093, 45.524517 ], [ 9.55028699799999, 45.52451699400001 ], [ 9.54089550000009, 45.501213 ], [ 9.51919150000009, 45.504474500000072 ], [ 9.483908500000013, 45.44868450000007 ], [ 9.458803663000083, 45.458856756000046 ], [ 9.434227, 45.468815 ], [ 9.425219754000011, 45.455801898 ], [ 9.415724231000013, 45.442083363000108 ], [ 9.410875, 45.43507750100008 ], [ 9.412653721000083, 45.433571604 ], [ 9.4259875, 45.422283 ], [ 9.40983463500001, 45.398416538000049 ], [ 9.404863503000087, 45.39107150500007 ], [ 9.40486350000009, 45.39107150000001 ], [ 9.401411281, 45.392967487000078 ], [ 9.391223860000082, 45.39856250400004 ], [ 9.388916, 45.39983 ], [ 9.382759827000086, 45.39465493 ], [ 9.373100500000078, 45.386535 ], [ 9.364549138000086, 45.36817860900004 ], [ 9.357994333000079, 45.354108035000081 ], [ 9.352684880999988, 45.342710743000026 ], [ 9.350357086000088, 45.337713887 ], [ 9.347544639000091, 45.331676676000029 ], [ 9.338828, 45.312965500000075 ], [ 9.32759511500015, 45.314757641000028 ], [ 9.29483145399999, 45.319984891000047 ], [ 9.250889332000071, 45.32699559800011 ], [ 9.24123076699999, 45.328536565000036 ], [ 9.237368901999986, 45.329152703000034 ], [ 9.230009720000084, 45.330326817000071 ], [ 9.201728, 45.334839 ], [ 9.187354223000085, 45.317234767 ], [ 9.18205250200009, 45.310741499000073 ], [ 9.198068613000089, 45.293991135000027 ], [ 9.200757, 45.291179500000027 ], [ 9.193098916999986, 45.293321220000109 ], [ 9.178710066000093, 45.297345319000044 ], [ 9.169327906000092, 45.299969208000107 ], [ 9.129581500000086, 45.311085 ], [ 9.090742571000078, 45.307791095000027 ], [ 9.074731493000087, 45.306433206 ], [ 9.069916830000011, 45.306024878000073 ], [ 9.036425500000092, 45.303184500000043 ], [ 9.025973735000093, 45.317570846000081 ], [ 9.022849422000093, 45.321871312000042 ], [ 9.019847000999988, 45.326003999000108 ], [ 8.991673507000087, 45.322750001000031 ], [ 8.9916735, 45.32275 ], [ 9.010436344999988, 45.29346777 ], [ 9.018697500000087, 45.280575 ], [ 9.011897425000086, 45.277611426000078 ], [ 8.996539793000011, 45.270918340000037 ], [ 8.98294150000001, 45.264992 ], [ 8.953186572000078, 45.285542928000041 ], [ 8.944611500000093, 45.291465500000072 ], [ 8.944311199999987, 45.300531765000073 ], [ 8.943877500000014, 45.3136255 ], [ 8.928849269000011, 45.319322826000075 ], [ 8.92112471100009, 45.322251270000038 ], [ 8.920407372999989, 45.322523219 ], [ 8.907376058000011, 45.327463498000043 ], [ 8.8660765, 45.343120500000026 ], [ 8.860931349000083, 45.354376905000038 ], [ 8.856379482000079, 45.364335341000071 ], [ 8.847292119000088, 45.384216399000081 ], [ 8.84289650000008, 45.393833 ], [ 8.836652074999989, 45.404249655 ], [ 8.827146260000092, 45.420106807000082 ], [ 8.818613808, 45.434340242000076 ], [ 8.81831224600009, 45.434843293000029 ], [ 8.818220235000013, 45.434996782000042 ], [ 8.815475885000012, 45.439574777000075 ], [ 8.805480261000071, 45.456249005000075 ], [ 8.789063371000083, 45.483634887000107 ], [ 8.78882198000008, 45.484037563000072 ], [ 8.787609006000082, 45.486060990000027 ], [ 8.787609, 45.486061 ], [ 8.76088221100008, 45.494968554000081 ], [ 8.758649351000088, 45.495712726000107 ], [ 8.752114545000012, 45.497890658000074 ], [ 8.722836500000085, 45.507648500000073 ], [ 8.716850586000078, 45.526655221000027 ], [ 8.712982800000077, 45.538936376000038 ], [ 8.712275392, 45.541182566000032 ], [ 8.706878499999988, 45.558319 ], [ 8.725432981000012, 45.565258714000038 ], [ 8.767663233000093, 45.581053597000107 ], [ 8.782250500000089, 45.586509500000034 ], [ 8.79400169200008, 45.582638581000026 ], [ 8.809398662000092, 45.577566719 ], [ 8.827325605999988, 45.571661467000069 ], [ 8.837002, 45.568474 ], [ 8.865976502000081, 45.575943001000041 ], [ 8.87308083100001, 45.582350372 ], [ 8.88312408200008, 45.591408348000073 ], [ 8.898034439000071, 45.604855951000047 ], [ 8.916010426000071, 45.62106843600003 ], [ 8.929746578000078, 45.633457028 ], [ 8.936001957000087, 45.639098734000072 ], [ 8.936902253999989, 45.639910710000038 ], [ 8.93997, 45.642677500000048 ], [ 8.955139290000091, 45.631706787000041 ], [ 8.960312829000088, 45.627965188000076 ], [ 8.97406550000008, 45.618019 ], [ 8.973625129999988, 45.608399531000032 ], [ 8.972677, 45.58768850000007 ], [ 8.97476321100001, 45.586729668000032 ], [ 8.99650172599999, 45.576738546 ] ] ], [ [ [ 9.487857545000082, 45.162317775000076 ], [ 9.478234500000013, 45.1616515 ], [ 9.446012515000092, 45.182744232 ], [ 9.4385435, 45.1876335 ], [ 9.465761469000086, 45.19923344800003 ], [ 9.469374296000012, 45.200773189000074 ], [ 9.470658500000013, 45.2013205 ], [ 9.508576350999988, 45.187379284000031 ], [ 9.518225879000084, 45.183831452000049 ], [ 9.526747, 45.1806985 ], [ 9.530893553999988, 45.167485204 ], [ 9.531565500000085, 45.165344 ], [ 9.51216266099999, 45.16400059800003 ], [ 9.504914765000081, 45.163498772000082 ], [ 9.487857545000082, 45.162317775000076 ] ] ] ] } }'), sensor_location)
  """.stripMargin)
milan_sensor.createOrReplaceTempView("milan_sensor")
milan_sensor.select("NomeStazione").show(100, false)

// view type of sensor
milan_sensor.select("Tipologia").dropDuplicates().show()

// knn query
var milan_sensor_d  = spark.sql(
  """
    |SELECT NomeStazione, ST_Distance( ST_Transform(ST_Point(9.231427, 45.480520), 'epsg:4326', 'epsg:3857'), new_sensor_location) AS distance, Valore
    |FROM milan_sensor
    |WHERE Tipologia='Temperatura'
    |ORDER BY distance DESC
  """.stripMargin)
milan_sensor_d.createOrReplaceTempView("milan_sensor_d")
milan_sensor_d.select("NomeStazione","distance").dropDuplicates().show(100,false)


// calc avg of temperature
milan_sensor_d.filter("distance < 2000").createOrReplaceTempView("my_sensor")
spark.sql("select NomeStazione,avg(Valore) from my_sensor group by NomeStazione").show(false)

//?
import org.apache.spark.sql.functions._
dati.select(substring(col("Data"),0,10)).dropDuplicates().show(100, false)


// check the max and min temperature in Bergamo province


// { "type": "Feature", "properties": { "STAT_LEVL_": 3, "NUTS_ID": "ITC46", "SHAPE_Leng": 2.961523, "SHAPE_Area": 0.318422 }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 10.169557500000082, 46.057041 ], [ 10.21205270100009, 46.050631357000043 ], [ 10.228079500000092, 46.048214 ], [ 10.243717049000082, 46.035629345000046 ], [ 10.254736301000094, 46.026761365000027 ], [ 10.262198498000089, 46.02075600100008 ], [ 10.206187, 46.007427 ], [ 10.194987113000082, 45.999238488 ], [ 10.188107702999986, 45.994208782000044 ], [ 10.17285070300008, 45.983054015000079 ], [ 10.147527212999989, 45.964539390000027 ], [ 10.114832736000011, 45.940635655000079 ], [ 10.112689542000084, 45.93906871400003 ], [ 10.103135361000085, 45.93208341700003 ], [ 10.090633500000081, 45.922943 ], [ 10.102825462999988, 45.898255874000029 ], [ 10.109030002000083, 45.885692499000044 ], [ 10.124206955000091, 45.877278461 ], [ 10.149791498000013, 45.863094501 ], [ 10.120407471000078, 45.836835156 ], [ 10.105442274000012, 45.823461351000049 ], [ 10.090872297000089, 45.810440739000029 ], [ 10.081689603000086, 45.802234528000042 ], [ 10.077217136000087, 45.798237662000048 ], [ 10.061094500000081, 45.783829500000081 ], [ 10.061240846000089, 45.78068611900008 ], [ 10.062754937000079, 45.748164909000081 ], [ 10.06364973400008, 45.728945514 ], [ 10.065496, 45.689289500000029 ], [ 10.047619491000091, 45.685805611 ], [ 9.997825565999989, 45.676101451000079 ], [ 9.942157509999987, 45.665252502000044 ], [ 9.942157500000093, 45.6652525 ], [ 9.937418463000086, 45.659551115000028 ], [ 9.926568395000089, 45.646497745 ], [ 9.910512826000087, 45.627181802000081 ], [ 9.907169192999987, 45.623159183000041 ], [ 9.891605500000082, 45.604435 ], [ 9.882499880000012, 45.604150657000048 ], [ 9.85055150000008, 45.603153 ], [ 9.849314248000013, 45.59845667700003 ], [ 9.84533541, 45.583353924000079 ], [ 9.843070912000087, 45.57475841300004 ], [ 9.841380192000088, 45.568340831000029 ], [ 9.836038500000086, 45.548065 ], [ 9.847482637000013, 45.530916831000042 ], [ 9.851833774000085, 45.524396982000042 ], [ 9.870261871000082, 45.496783879000049 ], [ 9.876145283999989, 45.487968032000026 ], [ 9.886262104999986, 45.472808745 ], [ 9.89028950000008, 45.466774 ], [ 9.889618, 45.427334 ], [ 9.88961799399999, 45.427334002 ], [ 9.881064751999986, 45.431186814000029 ], [ 9.875500404000093, 45.433693277 ], [ 9.85386478, 45.443439052 ], [ 9.838512186000088, 45.450354633000046 ], [ 9.82767, 45.4552385 ], [ 9.810853, 45.452648 ], [ 9.80998814600008, 45.44391674100001 ], [ 9.808534530000088, 45.42924155700004 ], [ 9.807894, 45.422775 ], [ 9.769303506000085, 45.430164499000028 ], [ 9.769303500000092, 45.430164500000046 ], [ 9.740120522000012, 45.458466519000041 ], [ 9.733365490000011, 45.465017634000048 ], [ 9.715175, 45.482659 ], [ 9.699880353000083, 45.469628719000042 ], [ 9.697256058000079, 45.467392949000043 ], [ 9.673990554999989, 45.447571894000049 ], [ 9.661828, 45.43721 ], [ 9.636768800000084, 45.455230768 ], [ 9.608649, 45.475452500000046 ], [ 9.590695568000086, 45.470754622000044 ], [ 9.586030371000078, 45.469533879000039 ], [ 9.550788, 45.460312 ], [ 9.547349863000079, 45.465117491000029 ], [ 9.522659110000092, 45.499627814 ], [ 9.51919150000009, 45.504474500000043 ], [ 9.54089550000009, 45.501213 ], [ 9.55028699799999, 45.52451699400001 ], [ 9.550286999000093, 45.524517 ], [ 9.526755500000093, 45.542755 ], [ 9.532536091999987, 45.562064400000082 ], [ 9.533597257999986, 45.565609102000082 ], [ 9.538598, 45.582313500000026 ], [ 9.531432896000013, 45.591925946 ], [ 9.503207100999987, 45.629792660000049 ], [ 9.495852004000085, 45.639659995000045 ], [ 9.495852, 45.63966 ], [ 9.48790750100008, 45.651863395000049 ], [ 9.486445638000078, 45.654108936000029 ], [ 9.486214, 45.654464750000045 ], [ 9.476576, 45.669269500000041 ], [ 9.474597982000091, 45.670441704000041 ], [ 9.461516894999988, 45.678193763000081 ], [ 9.451008, 45.684421500000042 ], [ 9.449994655000012, 45.69936404 ], [ 9.448857599000093, 45.71613079200003 ], [ 9.448236718000089, 45.725286151 ], [ 9.447815221000013, 45.731501452 ], [ 9.447642276000011, 45.73405165300008 ], [ 9.445847501000088, 45.760517 ], [ 9.45337036699999, 45.762946157000044 ], [ 9.470225359000011, 45.768388686 ], [ 9.492193723000014, 45.775482338000046 ], [ 9.51868049300009, 45.78403499800001 ], [ 9.518680499999988, 45.784035 ], [ 9.518680486000079, 45.784035018000083 ], [ 9.515935299000091, 45.787478622 ], [ 9.497380193999987, 45.810754432000039 ], [ 9.493752244000092, 45.815305388000041 ], [ 9.49203719799999, 45.817456768000028 ], [ 9.47518047599999, 45.838602101000049 ], [ 9.474183434999986, 45.839852805000049 ], [ 9.466533500000082, 45.849449 ], [ 9.47454055899999, 45.859592007 ], [ 9.48002176899999, 45.866535373 ], [ 9.495205346000091, 45.885769292000049 ], [ 9.505315802000013, 45.898576793000046 ], [ 9.505737761000091, 45.89911131400001 ], [ 9.511433218000093, 45.906326080000042 ], [ 9.540207, 45.942775500000039 ], [ 9.520086103000011, 45.965172522000046 ], [ 9.519911874000087, 45.965366460000041 ], [ 9.512701442000093, 45.973392554000043 ], [ 9.486097500000085, 46.003006 ], [ 9.526899500000013, 46.011509 ], [ 9.53271459700008, 46.012708912000079 ], [ 9.587315499999988, 46.0239755 ], [ 9.593233532999989, 46.027769062 ], [ 9.612571331000083, 46.040164927 ], [ 9.62348312200001, 46.047159576000041 ], [ 9.643457495000092, 46.059963497000041 ], [ 9.643457500000011, 46.0599635 ], [ 9.669896566000091, 46.060762056000044 ], [ 9.709259615000093, 46.061950963000044 ], [ 9.719439623999989, 46.062258436 ], [ 9.723824001000082, 46.062390860000079 ], [ 9.74575291700009, 46.06305319300003 ], [ 9.764773878000085, 46.063627695 ], [ 9.770131, 46.063789500000041 ], [ 9.770131008000078, 46.063789498 ], [ 9.800223326000094, 46.054824420000045 ], [ 9.804972486999986, 46.053409554000041 ], [ 9.808706333000089, 46.05229717 ], [ 9.812030424999989, 46.051306859000078 ], [ 9.81692522099999, 46.049848605000079 ], [ 9.836714500000085, 46.043953 ], [ 9.90806, 46.046375500000039 ], [ 9.940820500000086, 46.066604500000039 ], [ 9.983248717, 46.074471053000082 ], [ 10.011681195000079, 46.07974267600008 ], [ 10.041730347000083, 46.085314044000029 ], [ 10.053621385000014, 46.087518744000079 ], [ 10.058947479000011, 46.088506247000026 ], [ 10.066612989000078, 46.089927498 ], [ 10.066613002000082, 46.089927500000044 ], [ 10.102901500000087, 46.088825 ], [ 10.102836215000082, 46.080705891 ], [ 10.102817365000078, 46.078361707000028 ], [ 10.102607784000014, 46.052297646000028 ], [ 10.102595500000092, 46.05077 ], [ 10.169557500000082, 46.057041 ] ] ] } }

var bergamo_sensor = spark.sql(
  """
    |SELECT *
    |FROM spatialdf
    |WHERE ST_Contains (ST_GeomFromGeoJSON('{ "type": "Feature", "properties": { "STAT_LEVL_": 3, "NUTS_ID": "ITC46", "SHAPE_Leng": 2.961523, "SHAPE_Area": 0.318422 }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 10.169557500000082, 46.057041 ], [ 10.21205270100009, 46.050631357000043 ], [ 10.228079500000092, 46.048214 ], [ 10.243717049000082, 46.035629345000046 ], [ 10.254736301000094, 46.026761365000027 ], [ 10.262198498000089, 46.02075600100008 ], [ 10.206187, 46.007427 ], [ 10.194987113000082, 45.999238488 ], [ 10.188107702999986, 45.994208782000044 ], [ 10.17285070300008, 45.983054015000079 ], [ 10.147527212999989, 45.964539390000027 ], [ 10.114832736000011, 45.940635655000079 ], [ 10.112689542000084, 45.93906871400003 ], [ 10.103135361000085, 45.93208341700003 ], [ 10.090633500000081, 45.922943 ], [ 10.102825462999988, 45.898255874000029 ], [ 10.109030002000083, 45.885692499000044 ], [ 10.124206955000091, 45.877278461 ], [ 10.149791498000013, 45.863094501 ], [ 10.120407471000078, 45.836835156 ], [ 10.105442274000012, 45.823461351000049 ], [ 10.090872297000089, 45.810440739000029 ], [ 10.081689603000086, 45.802234528000042 ], [ 10.077217136000087, 45.798237662000048 ], [ 10.061094500000081, 45.783829500000081 ], [ 10.061240846000089, 45.78068611900008 ], [ 10.062754937000079, 45.748164909000081 ], [ 10.06364973400008, 45.728945514 ], [ 10.065496, 45.689289500000029 ], [ 10.047619491000091, 45.685805611 ], [ 9.997825565999989, 45.676101451000079 ], [ 9.942157509999987, 45.665252502000044 ], [ 9.942157500000093, 45.6652525 ], [ 9.937418463000086, 45.659551115000028 ], [ 9.926568395000089, 45.646497745 ], [ 9.910512826000087, 45.627181802000081 ], [ 9.907169192999987, 45.623159183000041 ], [ 9.891605500000082, 45.604435 ], [ 9.882499880000012, 45.604150657000048 ], [ 9.85055150000008, 45.603153 ], [ 9.849314248000013, 45.59845667700003 ], [ 9.84533541, 45.583353924000079 ], [ 9.843070912000087, 45.57475841300004 ], [ 9.841380192000088, 45.568340831000029 ], [ 9.836038500000086, 45.548065 ], [ 9.847482637000013, 45.530916831000042 ], [ 9.851833774000085, 45.524396982000042 ], [ 9.870261871000082, 45.496783879000049 ], [ 9.876145283999989, 45.487968032000026 ], [ 9.886262104999986, 45.472808745 ], [ 9.89028950000008, 45.466774 ], [ 9.889618, 45.427334 ], [ 9.88961799399999, 45.427334002 ], [ 9.881064751999986, 45.431186814000029 ], [ 9.875500404000093, 45.433693277 ], [ 9.85386478, 45.443439052 ], [ 9.838512186000088, 45.450354633000046 ], [ 9.82767, 45.4552385 ], [ 9.810853, 45.452648 ], [ 9.80998814600008, 45.44391674100001 ], [ 9.808534530000088, 45.42924155700004 ], [ 9.807894, 45.422775 ], [ 9.769303506000085, 45.430164499000028 ], [ 9.769303500000092, 45.430164500000046 ], [ 9.740120522000012, 45.458466519000041 ], [ 9.733365490000011, 45.465017634000048 ], [ 9.715175, 45.482659 ], [ 9.699880353000083, 45.469628719000042 ], [ 9.697256058000079, 45.467392949000043 ], [ 9.673990554999989, 45.447571894000049 ], [ 9.661828, 45.43721 ], [ 9.636768800000084, 45.455230768 ], [ 9.608649, 45.475452500000046 ], [ 9.590695568000086, 45.470754622000044 ], [ 9.586030371000078, 45.469533879000039 ], [ 9.550788, 45.460312 ], [ 9.547349863000079, 45.465117491000029 ], [ 9.522659110000092, 45.499627814 ], [ 9.51919150000009, 45.504474500000043 ], [ 9.54089550000009, 45.501213 ], [ 9.55028699799999, 45.52451699400001 ], [ 9.550286999000093, 45.524517 ], [ 9.526755500000093, 45.542755 ], [ 9.532536091999987, 45.562064400000082 ], [ 9.533597257999986, 45.565609102000082 ], [ 9.538598, 45.582313500000026 ], [ 9.531432896000013, 45.591925946 ], [ 9.503207100999987, 45.629792660000049 ], [ 9.495852004000085, 45.639659995000045 ], [ 9.495852, 45.63966 ], [ 9.48790750100008, 45.651863395000049 ], [ 9.486445638000078, 45.654108936000029 ], [ 9.486214, 45.654464750000045 ], [ 9.476576, 45.669269500000041 ], [ 9.474597982000091, 45.670441704000041 ], [ 9.461516894999988, 45.678193763000081 ], [ 9.451008, 45.684421500000042 ], [ 9.449994655000012, 45.69936404 ], [ 9.448857599000093, 45.71613079200003 ], [ 9.448236718000089, 45.725286151 ], [ 9.447815221000013, 45.731501452 ], [ 9.447642276000011, 45.73405165300008 ], [ 9.445847501000088, 45.760517 ], [ 9.45337036699999, 45.762946157000044 ], [ 9.470225359000011, 45.768388686 ], [ 9.492193723000014, 45.775482338000046 ], [ 9.51868049300009, 45.78403499800001 ], [ 9.518680499999988, 45.784035 ], [ 9.518680486000079, 45.784035018000083 ], [ 9.515935299000091, 45.787478622 ], [ 9.497380193999987, 45.810754432000039 ], [ 9.493752244000092, 45.815305388000041 ], [ 9.49203719799999, 45.817456768000028 ], [ 9.47518047599999, 45.838602101000049 ], [ 9.474183434999986, 45.839852805000049 ], [ 9.466533500000082, 45.849449 ], [ 9.47454055899999, 45.859592007 ], [ 9.48002176899999, 45.866535373 ], [ 9.495205346000091, 45.885769292000049 ], [ 9.505315802000013, 45.898576793000046 ], [ 9.505737761000091, 45.89911131400001 ], [ 9.511433218000093, 45.906326080000042 ], [ 9.540207, 45.942775500000039 ], [ 9.520086103000011, 45.965172522000046 ], [ 9.519911874000087, 45.965366460000041 ], [ 9.512701442000093, 45.973392554000043 ], [ 9.486097500000085, 46.003006 ], [ 9.526899500000013, 46.011509 ], [ 9.53271459700008, 46.012708912000079 ], [ 9.587315499999988, 46.0239755 ], [ 9.593233532999989, 46.027769062 ], [ 9.612571331000083, 46.040164927 ], [ 9.62348312200001, 46.047159576000041 ], [ 9.643457495000092, 46.059963497000041 ], [ 9.643457500000011, 46.0599635 ], [ 9.669896566000091, 46.060762056000044 ], [ 9.709259615000093, 46.061950963000044 ], [ 9.719439623999989, 46.062258436 ], [ 9.723824001000082, 46.062390860000079 ], [ 9.74575291700009, 46.06305319300003 ], [ 9.764773878000085, 46.063627695 ], [ 9.770131, 46.063789500000041 ], [ 9.770131008000078, 46.063789498 ], [ 9.800223326000094, 46.054824420000045 ], [ 9.804972486999986, 46.053409554000041 ], [ 9.808706333000089, 46.05229717 ], [ 9.812030424999989, 46.051306859000078 ], [ 9.81692522099999, 46.049848605000079 ], [ 9.836714500000085, 46.043953 ], [ 9.90806, 46.046375500000039 ], [ 9.940820500000086, 46.066604500000039 ], [ 9.983248717, 46.074471053000082 ], [ 10.011681195000079, 46.07974267600008 ], [ 10.041730347000083, 46.085314044000029 ], [ 10.053621385000014, 46.087518744000079 ], [ 10.058947479000011, 46.088506247000026 ], [ 10.066612989000078, 46.089927498 ], [ 10.066613002000082, 46.089927500000044 ], [ 10.102901500000087, 46.088825 ], [ 10.102836215000082, 46.080705891 ], [ 10.102817365000078, 46.078361707000028 ], [ 10.102607784000014, 46.052297646000028 ], [ 10.102595500000092, 46.05077 ], [ 10.169557500000082, 46.057041 ] ] ] } }'), sensor_location)
  """.stripMargin)
bergamo_sensor.createOrReplaceTempView("bergamo_sensor")
bergamo_sensor.select("NomeStazione").show(100, false)

spark.sql("select min(Valore),Max(Valore) from bergamo_sensor where Tipologia='Temperatura'").show(false)