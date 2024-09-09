import { observer } from "mobx-react-lite";
import langService from "../service/langService";
import toggle from "../assets/swap.svg";
import Button from "./Button";
import { useState } from "react";

const LangButt = observer(() => {
    const [isActive, setIsActive] = useState(false);

    const dynamicStyle = {
        transform: isActive ? 'rotateY(180deg)' : 'rotateY(0deg)',
        transition: 'transform 1s ease',
        height: '25px'
        
      };

    const tap = ()=>{
        langService.toggleLang();
        setIsActive(!isActive);
    }    
    return(
        <div style={dynamicStyle}>
            <Button img={toggle} tap={tap}  alt={"<->"}/>
        </div>
    );
});

export default LangButt;