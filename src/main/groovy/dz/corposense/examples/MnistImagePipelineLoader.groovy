package dz.corposense.examples

import groovy.transform.CompileStatic
import groovy.util.logging.Slf4j
import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.io.ClassPathResource

import javax.swing.JFileChooser

@Slf4j
@CompileStatic
class MnistImagePipelineLoader {

    static String fileChoose(){
        JFileChooser fc = new JFileChooser()
        if (fc.showOpenDialog(null) == JFileChooser.APPROVE_OPTION){
            return fc.selectedFile.absolutePath
        }
        return ""
    }

    static void main(String[] args) {
        int height = 28
        int width = 28
        int channels = 1
        List labelList = (1..9).toList()

        String fileChoose = fileChoose()
        if (fileChoose.isEmpty())
            return

        // Load trained model
        def trained_model_file = new ClassPathResource("mnist_png/trained_mnist_model.zip").file

        if ( !trained_model_file.exists() ){
            println("File 'trained_mnist_model.zip' is not found, please run 'runMnistImagePipelineExample' first.")
            return
        }

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(trained_model_file)

        log.info('**** Test image against trained network ****')

        def file = new File(fileChoose)

        NativeImageLoader loader = new NativeImageLoader(height, width, channels)

        INDArray image = loader.asMatrix(file)

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1)
        scaler.transform(image)

        INDArray output = model.output(image)

        log.info('*** file chosen: '+ fileChoose)
        log.info(output.toString())
        log.info(labelList.toString())

    }

}
