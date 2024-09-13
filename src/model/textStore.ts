
interface ITextStore {    
    getInput(): string; 
    getOutput(): string; 
    setText(input: string, output: string): void; 
    setInput(input: string): void;
    setOutput(output: string): void;
}

export default ITextStore;




