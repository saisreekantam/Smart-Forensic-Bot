#!/usr/bin/env node

console.log('Starting Vite debug script...');

try {
  import('vite').then(({ createServer }) => {
    import('path').then(path => {
      console.log('Current directory:', process.cwd());
      console.log('Vite config path:', path.resolve(process.cwd(), 'vite.config.ts'));
      
      createServer({
        configFile: path.resolve(process.cwd(), 'vite.config.ts'),
        logLevel: 'info',
        server: {
          host: 'localhost',
          port: 3001,
          strictPort: false
        }
      }).then(server => {
        console.log('Vite server created successfully');
        return server.listen();
      }).then(server => {
        console.log('Vite server started successfully');
        server.printUrls();
        console.log('Server is running...');
      }).catch(error => {
        console.error('Error starting Vite server:', error);
        console.error('Stack trace:', error.stack);
      });
    }).catch(error => {
      console.error('Error importing path:', error);
    });
  }).catch(error => {
    console.error('Error importing Vite:', error);
    console.error('Stack trace:', error.stack);
  });
  
} catch (error) {
  console.error('Unexpected error:', error);
  console.error('Stack trace:', error.stack);
}