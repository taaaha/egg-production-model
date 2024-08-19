document.addEventListener('DOMContentLoaded', () => {
        const formElement = document.getElementById('prediction-form');
        if (formElement) {
          formElement.addEventListener('submit', async (e) => {
            e.preventDefault();
      
            const data = [
              parseFloat(document.getElementById('Amount_of_chicken').value),
              parseFloat(document.getElementById('Amount_of_Feeding').value),
              parseFloat(document.getElementById('Ammonia').value),
              parseFloat(document.getElementById('Temperature').value),
              parseFloat(document.getElementById('Humidity').value),
              parseFloat(document.getElementById('Light_Intensity').value),
              parseFloat(document.getElementById('Noise').value)
            ];
      
            if (!validateInputs(data)) {
              alert('Please enter valid input values.');
              return;
            }
      
            try {
              // Show progress bar
              showProgressBar();
              await new Promise(resolve => setTimeout(resolve, 6000)); // Wait for 6 seconds
      
              const model = await loadModel();
              const result = await predict(model, data);
              updateUIWithPrediction(result);
            } catch (error) {
              console.error('Error during prediction process:', error);
              alert('An error occurred while making the prediction. Please try again.');
            } finally {
              hideProgressBar();
            }
          });
        } else {
          console.error('Form element not found');
        }
      });
      
      function validateInputs(data) {
        const [chickens, feeding, ammonia, temperature, humidity, light, noise] = data;
        return chickens > 0 && chickens <= 10000 &&
               feeding >= 50 && feeding <= 300 &&
               ammonia >= 0 && ammonia <= 50 &&
               temperature >= 0 && temperature <= 50 &&
               humidity >= 0 && humidity <= 100 &&
               light >= 0 && light <= 20000 &&
               noise >= 0 && noise <= 300;
      }
      
      async function loadModel() {
        try {
          console.log('Loading model from:', 'https://taaaha.github.io/egg-production-model/model.json');
          const model = await tf.loadLayersModel('https://taaaha.github.io/egg-production-model/model.json');
          return model;
        } catch (error) {
          console.error('Error loading model:', error);
          throw error;
        }
      }
      
      async function predict(model, data) {
        try {
          const sample = tf.tensor2d([data]);
      
          const csvData = await loadCSVData('https://taaaha.github.io/egg-production-model/dataset.csv');
          const { mean, stdDev } = preprocessData(csvData);
      
          const normalizedSample = sample.sub(mean).div(stdDev);
          const prediction = model.predict(normalizedSample);
          const result = prediction.dataSync()[0];
      
          return result;
        } catch (error) {
          console.error('Error during prediction:', error);
          throw error;
        }
      }
      
      async function loadCSVData(filePath) {
        try {
          const response = await fetch(filePath);
          const text = await response.text();
          const rows = text.split('\n').slice(1).filter(row => row.trim() !== '');
          const data = rows.map(row => {
            const values = row.split(',');
            return {
              Amount_of_chicken: parseFloat(values[0]),
              Amount_of_Feeding: parseFloat(values[1]),
              Ammonia: parseFloat(values[2]),
              Temperature: parseFloat(values[3]),
              Humidity: parseFloat(values[4]),
              Light_Intensity: parseFloat(values[5]),
              Noise: parseFloat(values[6]),
              Total_egg_production: parseFloat(values[7])
            };
          });
          return data;
        } catch (error) {
          console.error('Error loading CSV data:', error);
          throw error;
        }
      }
      
      function preprocessData(data) {
        const featureKeys = ['Amount_of_chicken', 'Amount_of_Feeding', 'Ammonia', 'Temperature', 'Humidity', 'Light_Intensity', 'Noise'];
        const targetKey = 'Total_egg_production';
      
        const features = data.map(row => featureKeys.map(key => row[key]));
        const targets = data.map(row => row[targetKey]);
      
        const featureTensor = tf.tensor2d(features);
        const targetTensor = tf.tensor2d(targets, [targets.length, 1]);
      
        const { mean, variance } = tf.moments(featureTensor, 0);
        const stdDev = variance.sqrt();
      
        return { mean, stdDev };
      }
      
      function showProgressBar() {
        const progressContainer = document.getElementById('progress-container');
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
      
        if (progressContainer) {
          progressContainer.style.display = 'block';
          let progress = 0;
          const interval = setInterval(() => {
            progress += 1;
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `Please wait... ${progress}%`;
      
            if (progress >= 100) {
              clearInterval(interval);
              setTimeout(() => {
                progressContainer.style.display = 'none';
              }, 100); // Short delay before hiding the progress container
            }
          }, 60); // Update progress every 60 milliseconds
        }
      }
      
      function hideProgressBar() {
        const progressContainer = document.getElementById('progress-container');
        if (progressContainer) {
          progressContainer.style.display = 'none';
        }
      }
      
      function updateUIWithPrediction(result) {
        const resultPopup = document.getElementById('result-popup');
        const resultText = document.getElementById('result-text');
      
        if (resultPopup && resultText) {
          resultText.textContent = `Predicted Egg Production: ${result.toFixed(2)} eggs`;
          resultPopup.style.display = 'block';
        }
      }
      
      const closePopupButton = document.getElementById('close-popup');
      if (closePopupButton) {
        closePopupButton.addEventListener('click', () => {
          document.getElementById('result-popup').style.display = 'none';
        });
      }
      
