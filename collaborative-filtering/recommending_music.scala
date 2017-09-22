/*
 * Copyright 2017 Mauro Pelucchi
 *
 * See LICENSE file for further information.
 */


//
// bin/spark-shell --driver-memory=2g
// 
// Recommending Music and the Audioscrobbler Data Set
//
// Data from http://www-etud.iro.umontreal.ca/%7Ebergstrj/audioscrobbler_data.html
// 

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD



val rawUserArtistData = sc.textFile("user_artist_data.txt")
rawUserArtistData.count


// print stat of the data
rawUserArtistData.map(_.split(' ')(0).toDouble).stats()
rawUserArtistData.map(_.split(' ')(1).toDouble).stats()


val rawArtistData = sc.textFile("artist_data.txt")
// get the list of id from artist dataset
val artistById = rawArtistData.flatMap{ line => val(id, name) = line.span(_ != '\t')
if (name.isEmpty) { None } else { try { Some((id.toInt, name.trim)) } catch { case e: NumberFormatException => None } } }

// show the first 5 values
artistById.take(5).foreach(println)

/*
(1134999,06Crazy Life)
(6821360,Pang Nakarin)
(10113088,Terfel, Bartoli- Mozart: Don)
(10151459,The Flaming Sidebur)
(6826647,Bodenstandig 3000)

*/


// get he artist alias from the dataset
val rawArtistAlias = sc.textFile("artist_alias.txt")
// create a map with id artist, id alias map
val artistAlias = rawArtistAlias.flatMap{ line => val tokens = line.split('\t')
if (tokens(0).isEmpty) { None } else {Some((tokens(0).toInt, tokens(1).toInt)) } }.sortBy(_._1).collectAsMap
artistAlias.take(100).foreach(println)

// check
artistById.lookup(2144298).head
// String = 03 Melvins

artistById.lookup(1002398).head
// String = Melvins



// create the input data for the model
import org.apache.spark.mllib.recommendation._
val bartistAlias = sc.broadcast(artistAlias)
val trainData = rawUserArtistData.map{ line => val Array(userId, artistId, count) = line.split(" ").map(_.toInt)
val finalArtistId = bartistAlias.value.getOrElse(artistId, artistId)
Rating(userId, finalArtistId, count)
}
trainData.count

// build the first model
val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)



// spot recommendations
// extract all lines for the user 2093760
val rawArtistsForUser = rawUserArtistData.map(_.split(" ")).filter{ case Array(user,_,_) => user.toInt == 2093760}
rawArtistsForUser.take(100)

// collect artists from the user data
val existingProducts = rawArtistsForUser.map{ case Array(user, artist, _) => artist.toInt }.collect.toSet

artistById.filter{ case(id, name) => existingProducts.contains(id)}.values.collect.foreach(println)
/*
David Gray                                                                      
Blackalicious
Jurassic 5
The Saw Doctors
Xzibit

--> Mix of pop and hip.pop!!!! 
*/

val recommendations = model.recommendProducts(2093760,5)
val recommmendedProductIDs = recommendations.map(_.product).toSet
artistById.filter{ case(id, name) => recommmendedProductIDs.contains(id)}.values.collect.foreach(println)

/*
50 Cent                                                                         
Snoop Dogg
Jay-Z
2Pac
The Game

--> Uhmm.. punk, rock, ... ops...
*/



// calculate the area under the ROC Curve
import org.apache.spark.rdd._
import org.apache.spark.broadcast.Broadcast


// from https://github.com/sryza/aas
 def areaUnderCurve(
      positiveData: RDD[Rating],
      bAllItemIDs: Broadcast[Array[Int]],
      predictFunction: (RDD[(Int,Int)] => RDD[Rating])) = {
    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive", and map to tuples
    val positiveUserProducts = positiveData.map(r => (r.user, r.product))
    // Make predictions for each of them, including a numeric score, and gather by user
    val positivePredictions = predictFunction(positiveUserProducts).groupBy(_.user)

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other items, excluding those that are "positive" for the user.
    val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions {
      // mapPartitions operates on many (user,positive-items) pairs at once
      userIDAndPosItemIDs => {
        // Init an RNG and the item IDs set once for partition
        val random = new Random()
        val allItemIDs = bAllItemIDs.value
        userIDAndPosItemIDs.map { case (userID, posItemIDs) =>
          val posItemIDSet = posItemIDs.toSet
          val negative = new ArrayBuffer[Int]()
          var i = 0
          // Keep about as many negative examples per user as positive.
          // Duplicates are OK
          while (i < allItemIDs.size && negative.size < posItemIDSet.size) {
            val itemID = allItemIDs(random.nextInt(allItemIDs.size))
            if (!posItemIDSet.contains(itemID)) {
              negative += itemID
            }
            i += 1
          }
          // Result is a collection of (user,negative-item) tuples
          negative.map(itemID => (userID, itemID))
        }
      }
    }.flatMap(t => t)
    // flatMap breaks the collections above down into one big set of tuples

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user)

    // Join positive and negative by user
    positivePredictions.join(negativePredictions).values.map {
      case (positiveRatings, negativeRatings) =>
        // AUC may be viewed as the probability that a random positive item scores
        // higher than a random negative one. Here the proportion of all positive-negative
        // pairs that are correctly ranked is computed. The result is equal to the AUC metric.
        var correct = 0L
        var total = 0L
        // For each pairing,
        for (positive <- positiveRatings;
             negative <- negativeRatings) {
          // Count the correctly-ranked pairs
          if (positive.rating > negative.rating) {
            correct += 1
          }
          total += 1
        }
        // Return AUC: fraction of pairs ranked correctly
        correct.toDouble / total
    }.mean() // Return mean AUC over users
  }




def buildRatings(
      rawUserArtistData: RDD[String],
      bArtistAlias: Broadcast[Map[Int,Int]]) = {
    rawUserArtistData.map { line =>
      val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
      val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
      Rating(userID, finalArtistID, count)
    }
}



val allData = buildRatings(rawUserArtistData, bartistAlias)
allData.count

// create train set and validation set
val Array(trainData, cvData) = allData.randomSplit(Array(0.7, 0.3))
trainData.cache
cvData.cache

trainData.count
cvData.count


val allItemIds = allData.map(_.product).distinct.collect
val ballItemIds = sc.broadcast(allItemIds)


// build new model
val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)
val auc = areaUnderCurve(cvData, ballItemIds, model.predict)

// about 0.5... is a good result?


// check most listened artists
  def predictMostListened(train: RDD[Rating])(allData: RDD[(Int,Int)]) = {
    val listenCount = train.map(r => (r.product, r.rating)).reduceByKey(_ + _).collectAsMap()
    allData.map { case (user, product) =>
      Rating(user, product, listenCount.getOrElse(product, 0.0))
    }
  }
val auc = areaUnderCurve(cvData, ballItemIds,predictMostListened(trainData))

// about 0.93... can we do better?

// model selection
//
 val evaluations =
      for (rank   <- Array(10,  50);
           lambda <- Array(1.0);
           alpha  <- Array(1.0, 40.0))
      yield {
        val model = ALS.trainImplicit(trainData, rank, 10, lambda, alpha)
        val auc = areaUnderCurve(cvData, ballItemIds, model.predict)
        ((rank, lambda, alpha), auc)
      }

evaluations.sortBy(_._2).reverse.foreach(println)


// best model --> 10, 1.0, 40

// ((10,1.0,40.0),0.9740943249304835)
// ((50,1.0,40.0),0.9724614833916505)
// ((10,1.0,1.0),0.9652102473589002)
// ((50,1.0,1.0),0.9597890685676553)

// making recommendations
val model = ALS.trainImplicit(allData, 50, 10, 1.0, 40.0)

val userID = 2093760
val recommendations = model.recommendProducts(userID, 5)
val recommendedProductIDs = recommendations.map(_.product).toSet
artistById.filter{ case(id, name) => recommendedProductIDs.contains(id)}.values.collect.foreach(println)
/*
50 Cent                                                                         
Snoop Dogg
Jay-Z
2Pac
Eminem
*/


val someUsers = allData.map(_.user).distinct().take(100)
val someRecommendations = someUsers.map(userID => model.recommendProducts(userID, 5))
someRecommendations.map( recs => recs.head.user + " -> " + recs.map(_.product).mkString(", ")).foreach(println)
someRecommendations.map( recs => recs.head.user + " -> " +  artistById.filter{ case(id, name) => recs.map(_.product).toSet.contains(id)}.values.collect.mkString(", ")).foreach(println)


/*

2188602 -> Esthero, Aim, Mandalay, Bliss, Olive                                 
2402725 -> 50 Cent, Jay-Z, Ludacris, 2Pac, The Game
2277899 -> [unknown], The Beatles, U2, Nirvana, Green Day
2003742 -> Haujobb, Rasputina, Seabound, The Birthday Massacre, Claire Voyant
1010282 -> The American Analog Set, Of Montreal, The Mountain Goats, Stars, The Wrens
2278380 -> [unknown], Juanes, Eminem, Avril Lavigne, Britney Spears
2091791 -> Joss Stone, Michael Bublé, James Horner, Jan A.P. Kaczmarek, Carlos Santana
2183961 -> Lindsay Lohan, Jesse McCartney, Skye Sweetnam, Hilary Duff, Ryan Cabrera
2073149 -> Mediæval Bæbes, The Tea Party, Vienna Teng, Tapping The Vein, Miranda Sex Garden
2094547 -> LeAnn Rimes, Phil Collins, Bryan Adams, Shania Twain, Céline Dion
1040429 -> Crosby, Stills, Nash & Young, Robert Johnson, Neil Young & Crazy Horse, The Band, The Byrds
1071850 -> Man Parrish, Op:l Bastards, Kojak, Rob Hubbard, The Beloved
2261584 -> Lateef & the Chief, West Indian Girl, Ellie Lawson, The Kleptones, Kinky
2156947 -> Namco Sound Team, Shadow Huntaz, aphilas, Nullsleep, DJ Assault
2052128 -> Kid Koala, Serge Gainsbourg, Funkadelic, CORNELIUS, Money Mark
2384837 -> Kamelot, Avantasia, Edguy, Ayreon, Demons & Wizards
2085122 -> Black Sabbath, Iron Maiden, Marillion, Porcupine Tree, John Petrucci
2248064 -> Yellowcard, Sum 41, My Chemical Romance, The Used, blink-182
2272855 -> Pinback, Q and Not U, Of Montreal, Mates of State, The Unicorns
2337894 -> Mae, Anberlin, Copeland, Further Seems Forever, Straylight Run
2079597 -> Alpha & Omega, Mafia, The Author, Eskimo, Orion
2417974 -> [unknown], Orishas, Eminem, Various Artists, Seeed
1023945 -> Kaki King, Unknown, David Cross, MC Frontalot, Bill Hicks
2306941 -> Pedro the Lion, [unknown], Death Cab for Cutie, Bright Eyes, Hot Hot Heat
2010489 -> The Mercury Program, Hilmar Örn Hilmarsson & Sigur Rós, Akira Yamaoka, Unknown, Final Fantasy
2090894 -> Tina Turner, Aretha Franklin, Carpenters, Bruce Springsteen & The E Street Band, Chris Rea
2028611 -> 2raumwohnung, Mediengruppe Telekommander, gustav, Mia, Quarks
2243371 -> K-OS, Pepper, Slightly Stoopid, Garden State, Saul Williams
2360800 -> Christian Walz, United Nations, Bodies Without Organs, Melody Club, Uniting Nations
2147405 -> Herb Alpert and The Tijuana Brass, Weather Report, Misc, Herbert Von Karajan, De Facto
2081170 -> [unknown], The Postal Service, The Beatles, Radiohead, Weezer
1005446 -> The Jimi Hendrix Experience, Crosby, Stills, Nash & Young, Grateful Dead, Neil Young & Crazy Horse, The Band
2066493 -> JOJO, Hilary Duff, Ryan Cabrera, Maria Mena, Daniel Bedingfield
2406352 -> Embrace, The Coral, Delays, Kaiser Chiefs, 22-20s
2093507 -> RX Bandits, Rufio, Rise Against, Tsunami Bomb, No Use for a Name
2247960 -> Ulrich Schnauss, Akufen, Two Lone Swordsmen, Plastikman, Ellen Allien
2247583 -> The Tiger Lillies, Future Bible Heroes, Sham 69, Hair, Tom Lehrer
1030887 -> Tom Novy, No Angels, B3, Galleon, Hardy Hard
2189837 -> Dj Encore, Oceanlab, Tiësto, 4 Strings, DJ Doboy
2270749 -> Ulver, The Gathering, Spinvis, Vive La Fête, SOPHIA
2351700 -> Antonio Vivaldi, ABBA, The Alan Parsons Project, Jean-Michel Jarre, Mike Oldfield
2094222 -> The Zutons, The Thrills, The Futureheads, Kaiser Chiefs, The Bravery
2040246 -> Paul McCartney & Wings, the pillows, Sweet, Spinal Tap, Cheap Trick
1048502 -> Gentleman, Jaya The Cat, Youngblood Brass Band, Ska-P, Tryo
2149004 -> [unknown], Eminem, U2, Nirvana, Green Day
2021825 -> Nicole, Viikate, Kotiteollisuus, Mokoma, To/Die/For
1025778 -> Mastodon, Suicidal Tendencies, Beatallica, Danzig, Anal Cunt
2039895 -> Lucero, The Anniversary, Slowreader, Heatmiser, Travis Morrison
1030939 -> Taproot, Adema, Soil, Sevendust, Stone Sour
2043119 -> Nightwish, Within Temptation, [unknown], Sonata Arctica, Pain
2229955 -> Mark Lanegan, Douglas Adams, dEUS, Mansun, My Vitriol
2326584 -> Fünf Sterne Deluxe, Tape, Verschiedene, Die Drei ???, Turntablerocker
2119117 -> Guided by Voices, Los Hermanos, French Kicks, Lori Meyers, A.C. Newman
2425176 -> the pillows, ASIAN KUNG-FU GENERATION, 増田俊郎, MC Chris, Andrew W.K.
1048424 -> Capitol K, Fog, The Soft Pink Truth, Hot Chip, Midnight Movies
2347553 -> Alison Krauss & Union Station, Jon Brion, Liz Phair, Nickel Creek, Nellie Mckay
2048540 -> Richard Gibbs, Chris Hülsbeck, Sonic Mayhem, Sean Callery, Pain
1059123 -> Insane Clown Posse, Group X, Raffi, CKY, Hatebreed
2195921 -> Mars Volta, Acoustic Alchemy, Sense Field, Candlebox, Chris Cornell
2088736 -> Chet Baker, Stan Getz & João Gilberto, Ben Harper & The Innocent Criminals, Astrud Gilberto, Israel Kamakawiwo'ole
2416791 -> [unknown], The Beatles, U2, Nirvana, Green Day


*/
