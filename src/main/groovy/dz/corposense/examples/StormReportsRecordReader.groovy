package dz.corposense.examples

import groovy.transform.CompileStatic
import groovy.transform.TypeChecked
import org.datavec.api.transform.DataAction
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema

@CompileStatic
@TypeChecked
class StormReportsRecordReader {

    static void main(String[] args) {

        String baseDir = '/datavec_spark_transform/'
        String fileName = 'reports.csv'
        String timeStamp = new Date().time.toString()
        String outputPath = baseDir+'reports_processed'+timeStamp

        /**
         * Data file looks like this
         * 161006-1655,UNK,2 SE BARTLETT,LABETTE,KS,37.03,-95.19,
         * TRAINED SPOTTER REPORTS TORNADO ON THE GROUND. (ICT),TOR
         * Fields are
         * datetime,severity,location,county,state,lat,lon,comment,type
         */

        Schema inputDataSchema = new Schema.Builder()
            .addColumnsString("datetime","severity","location","county","state","comment","type")
            .addColumnsDouble("lat","lon")
            .addColumnCategorical("type", "TOR", "WIND", "HAIL")
            .build()

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
            .removeColumns("datetime","severity","location","county","state","comment")
            .categoricalToInteger("type")
            .build()

        // Step through and print the before and after Schema
        tp.actionList.eachWithIndex{ DataAction action, int i ->
            println("(${action})--")
            println(tp.getSchemaAfterStep(i))
        }

        /**
         * Get our data into a spark RDD
         * and transform that spark RDD using our
         * transform process
         *

        SparkConf sparkConf = new SparkConf()
        sparkConf.setMaster("local[*]")
        sparkConf.setAppName("Storm Reports Record Reader Transform")
        JavaSparkContext sc = new JavaSparkContext(sparkConf)

        // read the data file
        JavaRDD<String> lines = sc.textFile(inputPath);
        // convert to Writable
        JavaRDD<List<Writable>> stormReports = lines.map(new StringToWritablesFunction(new CSVRecordReader()));
        // run our transform process
        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(stormReports,tp);
        // convert Writable back to string for export
        JavaRDD<String> toSave= processed.map(new WritablesToStringFunction(","));

        toSave.saveAsTextFile(outputPath);
        */



    }

}
