import { useEffect, useState } from "react";
import "./styles/keyboard.scss"
import { observer } from "mobx-react-lite";
import langStore from "../store/langStore";


interface KeyboardProps{
    get: () => string;
    set: (input:string) => void;
    esc: () => void;
}

const Keyboard = observer(({get, set, esc}:KeyboardProps) => {

    
    const [isCaps, setIsCaps] = useState<boolean>(false);
    const [isShift, setIsShift] = useState<boolean>(false);
    const [cursor, setCursor] = useState<number>(0);

    const textarea = document.querySelector('textarea');

    const keyboards:{[key: string]: string[][]} = {
        'ru': [
            [],
            ['Esc', '~.`', '!.1', '@.2', '#.3', '$.4', '%.5', '^.6', '&.7', '*.8', '(.9', ').0', '_.-', '+.=', 'Backspace'],
            ['Tab', 'ё', 'й', 'ц', 'у', 'к', 'е', 'н', 'г', 'ш', 'щ', 'з', 'х', 'ъ', '{_[', '}_]', '|_\\'],
            ['Caps Lock', 'ф', 'ы',  'в', 'а', 'п', 'р', 'о', 'л', 'д', 'ж', 'э',  ':_;', `"_'`, 'Enter'],
            ['Shift', 'я', 'ч', 'с', 'м', 'и', 'т', 'ь', 'б', 'ю',  '<_,', '>_.', '?_/', 'Shift'],
            ['Ctrl', 'Alt', ' ', 'Ctrl', 'Alt']
        ],

        'mans':[
            ['Esc', 'а̄', 'ē', 'ё̄', 'ӣ', 'ӈ', 'о̄', 'ӯ', 'ы̄', 'э̄', 'я̄', 'ю̄', 'Shift',],
            ['Esc', '~.`', '!.1', '@.2', '#.3', '$.4', '%.5', '^.6', '&.7', '*.8', '(.9', ').0', '_.-', '+.=', `".'`, 'Backspace'],
            ['Tab', 'ё', 'ё̄', 'й', 'ц', 'у', 'ӯ', 'к', 'е', 'ē', 'н', 'ӈ', 'г', 'ш', 'щ', 'з', 'х', 'ъ', '{_[', '}_]'],
            ['Caps Lock', 'ф', 'ы', 'ы̄', 'в', 'а', 'а̄', 'п', 'р', 'о', 'о̄', 'л', 'д', 'ж', 'э', 'э̄', ':_;', 'Enter'],
            ['Shift', 'я', 'я̄', 'ч', 'с', 'м', 'и', 'ӣ', 'т', 'ь', 'б', 'ю', 'ю̄', '<_,', '>_.', '?_/', 'Shift'],
            ['Ctrl', 'Alt', ' ', 'Ctrl', 'Alt'/*, '<', '>'*/]
        ]
    }
    console.log(keyboards[langStore.fromLang][1])
    useEffect(()=>{
        if(textarea){
            textarea.selectionStart = cursor;
            textarea.selectionEnd = cursor;
            textarea.focus();
        }
        
        
        
    }, [cursor, textarea]);
    
    const handleKeyClick = (key: string) => {
        if(textarea){
            textarea.focus();
        }

        if (key === 'Enter') {
            handleEnterKey();
        } 
        else if(key === "Ctrl" || key === "Alt" || key === '<' || key === '>')
        { /* empty */ } else if (key === ' ') {
            handleSpaceKey();
        } else if (key === 'Caps Lock') {
            handleCapsLock();
        } else if (key === 'Backspace') {
            handleDeleteKey();
        } else if (key === 'Shift') {
            handleShiftKey();
        } else if (key === 'Tab') {
            handleTabKey();
        } else if (key === 'Esc') {
            esc();
        } else {
            handleRegularKey(key);
        }
    };


    const handleSpaceKey = () => {
        const newContent = get() + '\u00A0';
        set(newContent);
    };

    const handleEnterKey = () => {
        const newContent = get() + '\n';
        set(newContent);
    };

    const handleCapsLock = () => {
        const updatedCaps = !isCaps;
        setIsCaps(updatedCaps);
        const keys = document.querySelectorAll('.key');
        keys.forEach((key) => {
            const firstSpanElement:HTMLSpanElement | null = key.querySelector('span:first-child');
            if (firstSpanElement) {
                const keyText = firstSpanElement.innerText.toLowerCase();
                if (!['shift', 'alt', 'ctrl', 'enter', 'caps lock', 'tab']
                    .includes(keyText)) {
                    firstSpanElement.innerText = 
                    ((updatedCaps && isShift) || (!updatedCaps && !isShift)) 
                    ? keyText.toLowerCase() : keyText.toUpperCase();
                }
                if (keyText === 'caps lock') {
                    firstSpanElement.parentElement!.style.backgroundColor = 
                    (updatedCaps) ? '#445760' : '#7A7A7A';
                }
            }
        });
    };

    const handleTabKey = () => {
        const newContent = get() + '    ';
        set(newContent);
    };

    const handleDeleteKey = () => {
        if(textarea){
            textarea.focus();
            let cursorPosition = textarea.selectionStart;
            const cursorPositionEnd = textarea.selectionEnd;
            const inputText = get();
            if (inputText.length === 0 || (cursorPosition === 0 && cursorPosition === cursorPositionEnd) ) {
                return;
            }
            if(cursorPosition < cursorPositionEnd){
                cursorPosition += 1;
            }
            const newContent = inputText.slice(0, cursorPosition - 1) + inputText.slice(cursorPositionEnd, inputText.length);
            textarea.selectionStart = cursorPosition - 1;
            textarea.selectionEnd = cursorPosition - 1;
            setCursor(cursorPosition - 1);
            set(newContent);
            
        }
        
    };

    const handleShiftKey = () => {
        const updatedShift = !isShift;
        setIsShift(updatedShift);
        const keys = document.querySelectorAll('.key');
        keys.forEach((key) => {
            const firstSpanElement:HTMLSpanElement | null = key.querySelector('span:first-child');
            if (firstSpanElement) {
                const keyText = firstSpanElement.innerText.toLowerCase();
                if (!['shift', 'alt', 'ctrl', 'enter', 'caps lock', 'tab', 'esc', 'backspace'].
                    includes(keyText)) {
                    firstSpanElement.innerText = 
                    ((updatedShift && isCaps) || (!updatedShift && !isCaps)) 
                    ? keyText.toLowerCase() : keyText.toUpperCase();
                }
                if (keyText === 'shift') {
                    firstSpanElement.parentElement!.style.backgroundColor = 
                    (updatedShift) ? '#445760' : '#7A7A7A';
                }
            }
        });
    };

    const handleRegularKey = (key: string) => {
        const inputText = get();
        const keys = key.split(/[._]/);
        let newContent: string;
        if (keys.length > 1) {
            if (isShift) {
                if (keys.length === 3) {
                    if (keys[0] === '>') newContent = inputText + '>';
                    else newContent = inputText + '_';
                }
                else newContent = inputText + keys[0];
            } else {
                if (keys.length === 3) {
                    if (keys[0] === '>') newContent = inputText + '.';
                    else newContent = inputText + '-';
                }
                else newContent = inputText + keys[1];
            }
        } else {
            const character = ((isShift && isCaps) || (!isShift && !isCaps)) 
            ? key.toLowerCase() : key.toUpperCase();
            newContent = inputText + character;
        }
        set(newContent);
    };


    return (
        <div className={`keyboard ${langStore.fromLang}`}>
            {/* <div className="textcontainer">
                <pre>{inputText}</pre>
            </div> */}
            <div className="keyboardcontainer">
                <div className="container">
                <div className="row row__mobile">
                        {keyboards[langStore.fromLang][0]
                        .map((keyvalue) => (
                            <div key={keyvalue} className='key' 
                                 onClick={() => handleKeyClick(keyvalue)}>
                                {keyvalue.includes('_') ? (
                                    keyvalue.split('_').map((part, index) => (
                                        <span key={index}>{part}</span>
                                    ))
                                ) : (
                                    <span>{keyvalue}</span>
                                )}
                            </div>
                        ))}
                    </div>
                    <div className="row">
                        {keyboards[langStore.fromLang][1]
                        .map((keyvalue) => 
                        (
                            <div key={keyvalue} className='key' 
                                 onClick={() => handleKeyClick(keyvalue)}>
                                {keyvalue.includes('.') ? (
                                    keyvalue.split('.').map((part, index) => (
                                        <span key={index}>{part}</span>
                                    ))
                                ) : (
                                    <span>{keyvalue}</span>
                                    // keyvalue === 
                                    //   'Backspace' 
                                    //  ? (
                                    //     <i className="fa-solid fa-delete-left"></i>
                                    // ) : (
                                    //     <span>{keyvalue}</span>
                                    // )
                                )}
                            </div>
                        ))}
                    </div>
                    
                    <div className="row">
                        {keyboards[langStore.fromLang][2]
                        .map((keyvalue) => (
                            <div key={keyvalue} className='key' 
                                 onClick={() => handleKeyClick(keyvalue)}>
                                {keyvalue.includes('_') ? (
                                    keyvalue.split('_').map((part, index) => (
                                        <span key={index}>{part}</span>
                                    ))
                                ) : (
                                    <span>{keyvalue}</span>
                                )}
                            </div>
                        ))}
                    </div>
                    <div className="row">
                        {keyboards[langStore.fromLang][3]
                            .map((keyvalue) => (
                            <div key={keyvalue} className='key' 
                                 onClick={() => handleKeyClick(keyvalue)}>
                                {keyvalue.includes('_') ? (
                                    keyvalue.split('_').map((part, index) => (
                                        <span key={index}>{part}</span>
                                    ))
                                ) : (
                                    <span>{keyvalue}</span>
                                )}
                            </div>
                        ))}
                    </div>
                    <div className="row">
                        {keyboards[langStore.fromLang][4].map((keyvalue, index) => (
                            <div key={index} className='key' 
                                 onClick={() => handleKeyClick(keyvalue)}>
                                {keyvalue.includes('_') ? (
                                    keyvalue.split('_').map((part, index) => (
                                        <span key={index}>{part}</span>
                                    ))
                                ) : (
                                    <span>{keyvalue}</span>
                                )}
                            </div>
                        ))}
                    </div>
                    <div className="row">
                        {keyboards[langStore.fromLang][5]
                            .map((keyvalue, index) => (
                            <div key={index} className='key' 
                            onClick={() => handleKeyClick(keyvalue)}>
                                <span>{keyvalue}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
});

export default Keyboard;