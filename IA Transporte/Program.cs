using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TransportAI
{
    class Program
    {
        static void Main(string[] args)
        {
            // Definir o contexto do ML.NET
            var context = new MLContext();

            // Carregar os dados de treinamento
            var data = context.Data.LoadFromTextFile<InputData>(
                "dados_treinamento.csv", separatorChar: ',');

            // Dividir os dados em conjuntos de treinamento e teste
            var trainTestSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

            // Pipeline para colunas que não são chave
            var nonKeyPipeline = context.Transforms.Concatenate("NonKeyFeatures",
                                    nameof(InputData.Distance), nameof(InputData.Weather),
                                    nameof(InputData.Traffic))
                                .Append(context.Transforms.Conversion.MapValueToKey(
                                    "Label", nameof(InputData.DeliveryTime)))
                                .Append(context.Transforms.NormalizeMinMax(
                                    "NonKeyFeatures", "NonKeyFeatures"))
                                .Append(context.Transforms.DropColumns(
                                    nameof(InputData.DeliveryTime), nameof(InputData.VehicleType),
                                    nameof(InputData.Weather), nameof(InputData.Traffic)));

            // Pipeline para colunas que são chave
            var keyPipeline = context.Transforms.Conversion.MapValueToKey(
                                nameof(InputData.VehicleTypeEncoded), nameof(InputData.VehicleType))
                            .Append(context.Transforms.Categorical.OneHotEncoding(
                                nameof(InputData.VehicleTypeEncoded)));

            // Unir as saídas dos dois pipelines
            var pipeline = context.Transforms.Concatenate("Features",
                                "NonKeyFeatures", nameof(InputData.VehicleTypeEncoded))
                            .Append(context.Transforms.CopyColumns("FeaturesEncoded", "Features")
                                .Append(context.Transforms.Categorical.OneHotEncoding(
                                    "FeaturesEncoded", nameof(InputData.VehicleTypeEncoded))))
                            .Append(context.Transforms.NormalizeMinMax(
                                "FeaturesEncoded", "FeaturesEncoded"))
                            .Append(context.Transforms.DropColumns(
                                nameof(InputData.VehicleTypeEncoded)));

            // Definir o modelo de aprendizado de máquina
            var trainer = context.BinaryClassification.Trainers.LightGbm();
            var trainingPipeline = pipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainTestSplit.TrainSet);


            // Avaliar a precisão do modelo
            var predictions = trainedModel.Transform(trainTestSplit.TestSet);
            var metrics = context.BinaryClassification.Evaluate(predictions);
            Console.WriteLine($"Precisão: {metrics.Accuracy}");

            // Salvar o modelo em um arquivo
            using (var stream = new FileStream("model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
                context.Model.Save(trainedModel, trainTestSplit.TrainSet.Schema, stream);


            // Carregar o modelo a partir do arquivo
            ITransformer loadedModel;
            using (var stream = new FileStream("model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = context.Model.Load(stream, out var modelSchema);

            // Realizar uma previsão com base nos dados de entrada
            var predictionEngine = context.Model.CreatePredictionEngine<InputData, OutputData>(loadedModel);
            var input = new InputData { Distance = 100, VehicleType = "car", Weather = "sunny", Traffic = "low" };
            var output = predictionEngine.Predict(input);
            Console.WriteLine($"Tempo de entrega previsto: {output.DeliveryTime}");
        }

    }

    // Definir o esquema de dados de entrada
    public class OutputData
    {
        [ColumnName("PredictedLabel")]
        public bool DeliveryTime;
    }

    public class InputData
    {
        [LoadColumn(0)]
        public float Distance { get; set; }

        [LoadColumn(1)]
        public string VehicleType { get; set; }

        [LoadColumn(2)]
        public string Weather { get; set; }

        [LoadColumn(3)]
        public string Traffic { get; set; }

        [LoadColumn(4)]
        public bool DeliveryTime { get; set; }
        [LoadColumn(5)]
        public float VehicleTypeEncoded { get; set; }
    }
}