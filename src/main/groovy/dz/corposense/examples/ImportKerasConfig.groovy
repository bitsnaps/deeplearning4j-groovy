package dz.corposense.examples

import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import groovy.transform.CompileStatic
import groovy.util.logging.Slf4j
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.io.ClassPathResource

@Slf4j
@CompileStatic
class ImportKerasConfig {

    static void main(String[] args) {

        MultiLayerNetwork model = KerasModelImport
                .importKerasSequentialModelAndWeights(new ClassPathResource('/iris/iris_model_save.h5').file.absolutePath)

        int numLinesToSkip = 0
        String delimiter = ','
        // Read the iris DataSet file as a collection of records
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
        recordReader.initialize(new FileSplit(new ClassPathResource('/iris/iris.csv').file))

        // label index
        int labelIndex = 4
        // num of classes
        int numClasses = 3
        // batchSize
        int batchSize = 150
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)
        DataSet allData = iterator.next()
        allData.shuffle()

        // Evaluate the model
        Evaluation eval = new Evaluation(3)
        INDArray output = model.output(allData.featureMatrix)
        eval.eval(allData.labels, output)
        log.info(eval.stats())

    }

}
